import copy
import os
from collections import deque
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from sqlmodel import Session, and_, col, not_, or_, select

from api.models import (
    DeleteDocumentRequestBody,
    DocumentInKnowledgeBase,
    GetDocumentStatusReponse,
    GetKnowledgeBase,
    GetKnowledgeBaseResponse,
    InheritKnowledgeBaseRequest,
    KnowledgeBaseRequest,
    KnowledgeBaseResponse,
    InheritableKnowledgeBaseResponse,
    UploadFileResponse,
    UserResponse,
)
from celery.result import AsyncResult
from database.core.s3 import get_s3_client
from src.celery import celery_app
from src.constants import ErrorResponse, FileStatus
from src.database import (
    Assistants,
    DatabaseManager,
    DocumentChunks,
    Documents,
    KnowledgeBases,
    S3Client,
    Users,
    get_db_manager,
    get_session,
    is_valid_uuid,
)
from src.settings import default_settings
from src.tasks import parse_document
from src.utils import get_formatted_logger

from .user_router import get_current_user

logger = get_formatted_logger(__file__)

kb_router = APIRouter()


UPLOAD_FOLDER = Path(default_settings.upload_temp_folder)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

DOWNLOAD_FOLDER = Path("downloads")
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


@kb_router.post("/create", response_model=KnowledgeBaseResponse)
async def create_new_knowledge_base(
    kb_info: Annotated[KnowledgeBaseRequest, Body(...)],
    db_session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[Users, Depends(get_current_user)],
):
    """
    Create new knowledge base
    """

    with db_session as session:
        kb = KnowledgeBases(
            name=kb_info.name,
            description=kb_info.description,
            user_id=current_user.users_id,
            is_contextual_rag=True,
        )

        session.add(kb)
        session.commit()
        session.refresh(kb)

        return KnowledgeBaseResponse(
            id=kb.knowledge_bases_id,
            name=kb.name,
            description=kb.description,
            created_at=kb.created_at,
            updated_at=kb.updated_at,
            user=UserResponse(
                id=current_user.users_id,
                username=current_user.username,
                created_at=current_user.created_at,
                updated_at=current_user.updated_at,
            ),
        )


@kb_router.post(
    "/upload",
    response_model=UploadFileResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Knowledge Base not found",
            "content": {
                "application/json": {"example": {"detail": "Knowledge Base not found"}}
            },
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid Knowledge Base ID",
            "content": {
                "application/json": {"example": {"detail": "Invalid Knowledge Base ID"}}
            },
        },
        409: {
            "model": ErrorResponse,
            "description": "File already exists in the Knowledge Base",
            "content": {
                "application/json": {
                    "example": {"detail": "File already exists in the Knowledge Base"}
                }
            },
        },
    },
)
async def upload_file(
    knowledge_base_id: str,
    file: Annotated[UploadFile, File(...)],
    db_session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[Users, Depends(get_current_user)],
    s3_client: Annotated[S3Client, Depends(get_s3_client)],
):
    """
    Upload file to knowledge base for the current user
    """

    if not is_valid_uuid(knowledge_base_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Knowledge Base ID",
        )

    file_name = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, file_name)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    if os.path.exists(file_path):
        logger.info("File đã được ghi thành công ở container!")
        with open(file_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
    else:
        logger.info("Ghi file thất bại!")

    with db_session as session:
        query = select(KnowledgeBases).where(
            KnowledgeBases.knowledge_bases_id == knowledge_base_id
        )

        kb = session.exec(query).first()

        query_documents = select(Documents).where(
            Documents.user_id == current_user.users_id,
        )

        documents = session.exec(query_documents).all()

        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge Base not found"
            )

        if not current_user.allow_upload(file_size, documents):
            logger.debug(
                f"Deleting the file from local as user has no space left for uploading files. Max size is {current_user.max_size_mb} MB"
            )
            Path(file_path).unlink()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"No space left for uploading files, used: {round(current_user.total_upload_size(documents=documents), 2)} MB. Allowed space: {current_user.max_size_mb} MB. This file size: {round(file_size / (1024 * 1024), 2)} MB",
            )

        if kb.user_id != current_user.users_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to upload to this Knowledge Base",
            )

        folder_path_in_s3 = os.path.join(
            default_settings.upload_bucket_name, knowledge_base_id
        )
        document = Documents(
            file_name=file.filename,
            file_path_in_minio=f"{file.filename}",
            file_type=Path(file.filename).suffix,
            status=FileStatus.UPLOADED,
            file_size=file_size,
            knowledge_base_id=knowledge_base_id,
            user_id=current_user.users_id,
        )

        session.add(document)
        session.commit()
        session.refresh(document)

        s3_client.upload_file(
            bucket_name=folder_path_in_s3,
            object_name=document.file_path_in_minio,
            file_path=str(file_path),
        )

        # Remove the file in local after uploading to Minio
        Path(file_path).unlink()

        return UploadFileResponse(
            doc_id=document.documents_id,
            file_name=document.file_name,
            file_type=document.file_type,
            status=document.status,
            knowledge_base=kb,
            created_at=document.created_at,
            file_size_in_mb=document.file_size_in_mb,
        )


@kb_router.post("/process/{document_id}/{top_k}")
async def process_document(
    document_id: str,
    top_k: str,
    db_session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[Users, Depends(get_current_user)],
):
    """
    Process document
    """
    with db_session as session:
        query = select(Documents).where(Documents.documents_id == document_id)

        document = session.exec(query).first()

        query_kb = select(KnowledgeBases).where(
            KnowledgeBases.knowledge_bases_id == document.knowledge_base_id
        )

        kb = session.exec(query_kb).first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found !"
            )

        if document.user_id != current_user.users_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to process this document",
            )

        is_contextual_rag = kb.is_contextual_rag

        isContextualRAG = is_contextual_rag

        folder_path_in_s3 = os.path.join(
            str(document.knowledge_base_id),
            document.file_path_in_minio,
        )

        task = parse_document.delay(
            folder_path_in_s3,
            document.documents_id,
            document.knowledge_base_id,
            isContextualRAG,
            int(top_k)
        )

        document.task_id = task.id
        document.status = FileStatus.PROCESSING

        session.add(document)
        session.commit()

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"task_id": task.id, "status": "Processing"},
        )


@kb_router.post("/stop_processing/{document_id}")
async def stop_processing_document(
    document_id: str,
    db_session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[Users, Depends(get_current_user)],
):
    """
    Stop processing document
    """
    with db_session as session:
        query = select(Documents).where(Documents.documents_id == document_id)

        document = session.exec(query).first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found !"
            )

        if document.user_id != current_user.users_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to stop processing this document",
            )

        task_id = document.task_id

        if not task_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Task ID not found !"
            )

        task = AsyncResult(task_id, app=celery_app)

        if task.state == "PROGRESS":
            task.revoke(terminate=True, signal="SIGKILL")

            document.status = FileStatus.FAILED

            session.add(document)
            session.commit()

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": "Processing stopped successfully"},
            )

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "Task is not in progress"},
        )


@kb_router.get("/document_status/{document_id}")
async def get_document_status(
    document_id: str,
    db_session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[Users, Depends(get_current_user)],
):
    with db_session as session:
        query = select(Documents).where(Documents.documents_id == document_id)

        document = session.exec(query).first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found !"
            )

        if document.user_id != current_user.users_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to access this document",
            )

        task_id = document.task_id

        if document.status != FileStatus.PROCESSING:
            return GetDocumentStatusReponse(
                doc_id=document.documents_id,
                status=document.status,
                task_id=task_id,
                metadata={},
            )

        if not task_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Task ID not found !"
            )

        task = AsyncResult(task_id, app=celery_app)

        state = task.state

        response = {
            "document_id": document.documents_id,
            "file_name": document.file_name,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
        }

        if state == "SUCCESS":
            response["status"] = FileStatus.PROCESSED
            response["metadata"] = {}
            document.status = FileStatus.PROCESSED

        elif state == "FAILURE":
            response["status"] = FileStatus.FAILED
            response["metadata"] = {}
            document.status = FileStatus.FAILED

        elif state == "PROGRESS":
            response["status"] = FileStatus.PROCESSING
            response["progress"] = task.info["progress"]
        else:
            response["status"] = state

        session.add(document)
        session.commit()
        return response


@kb_router.get("/get_all", response_model=list[GetKnowledgeBase])
async def get_all_knowledge_bases(
    db_session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[Users, Depends(get_current_user)],
):
    """
    Get all knowledge bases
    """
    with db_session as session:
        query = select(KnowledgeBases).where(
            KnowledgeBases.user_id == current_user.users_id
        )

        knowledge_bases = session.exec(query).all()

        documents = [
            session.exec(
                select(Documents).where(
                    Documents.knowledge_base_id == kb.knowledge_bases_id
                )
            ).all()
            for kb in knowledge_bases
        ]

        result = [
            GetKnowledgeBase(
                id=kb.knowledge_bases_id,
                name=kb.name,
                description=kb.description,
                document_count=len(document),
                last_updated=kb.last_updated,
            )
            for kb, document in zip(knowledge_bases, documents)
        ]

        session.close()

        return result


@kb_router.get("/{kb_id}", response_model=GetKnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: str,
    db_session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[Users, Depends(get_current_user)],
):
    """
    Get knowledge base by ID
    """
    with db_session as session:
        query = select(KnowledgeBases).filter_by(
            knowledge_bases_id=kb_id, user_id=current_user.users_id
        )

        kb = session.exec(query).first()

        query_documents = select(Documents).where(Documents.knowledge_base_id == kb_id)

        documents = session.exec(query_documents).all()

        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge Base not found !",
            )

        if kb.user_id != current_user.users_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to access this Knowledge Base",
            )

        if kb.parents:
            mergeable_kb = []
        else:
            mergeable_kb = session.exec(
                select(KnowledgeBases.knowledge_bases_id, KnowledgeBases.name)
                .join(Users, KnowledgeBases.user_id == Users.users_id)
                .where(
                    Users.organization == current_user.organization,
                    not_(
                        or_(
                            col(KnowledgeBases.parents).contains(
                                [kb.knowledge_bases_id]
                            ),
                            col(KnowledgeBases.children).contains(
                                [kb.knowledge_bases_id]
                            ),
                        ),
                    ),
                    not_(KnowledgeBases.knowledge_bases_id == kb.knowledge_bases_id),
                    and_(kb.children == []),
                )
            ).all()

        session.close()

        return GetKnowledgeBaseResponse(
            id=kb.knowledge_bases_id,
            name=kb.name,
            description=kb.description,
            user_id=kb.user_id,
            created_at=kb.created_at,
            updated_at=kb.updated_at,
            document_count=len(documents),
            last_updated=kb.last_updated,
            parents=kb.parents,
            children=kb.children,
            documents=[
                DocumentInKnowledgeBase(
                    id=doc.documents_id,
                    file_name=doc.file_name,
                    file_type=doc.file_type,
                    status=doc.status,
                    created_at=doc.created_at,
                    file_size_in_mb=doc.file_size_in_mb,
                )
                for doc in documents
            ],
            inheritable_knowledge_bases=(
                [
                    InheritableKnowledgeBaseResponse(id=kb.knowledge_bases_id, name=kb.name)
                    for kb in mergeable_kb
                ]
            ),
        )


@kb_router.get("/download/{document_id}")
async def download_document(
    document_id: str,
    db_session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[Users, Depends(get_current_user)],
    s3Client: Annotated[S3Client, Depends(get_s3_client)],
):
    """
    Download document
    """
    with db_session as session:
        query = select(Documents).where(Documents.documents_id == document_id)

        document = session.exec(query).first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found !"
            )

        if document.user_id != current_user.users_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to download this document",
            )

        file_path = DOWNLOAD_FOLDER / document.file_name

        folder_path_in_s3 = os.path.join(
            default_settings.upload_bucket_name, document.knowledge_base_id
        )

        s3Client.download_file(
            bucket_name=folder_path_in_s3,
            object_name=document.file_path_in_minio,
            file_path=str(file_path),
        )

        session.close()

        return FileResponse(path=file_path, filename=document.file_name)


@kb_router.delete("/delete_document/{document_id}")
async def delete_document(
    document_id: str,
    delete_document_request_body: DeleteDocumentRequestBody,
    db_session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[Users, Depends(get_current_user)],
    db_manager: Annotated[DatabaseManager, Depends(get_db_manager)],
):
    """
    Delete document
    """
    with db_session as session:
        query = select(Documents).where(Documents.documents_id == document_id)

        document = session.exec(query).first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found !"
            )

        if document.user_id != current_user.users_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to delete this document",
            )

        folder_path_in_s3 = os.path.join(
            default_settings.upload_bucket_name, document.knowledge_base_id
        )
        db_manager.delete_file(
            bucket_name=folder_path_in_s3,
            object_name=document.file_path_in_minio,
            delete_to_retry=delete_document_request_body.delete_to_retry,
            document_id=document.documents_id,
        )

        document_chunks = session.exec(
            select(DocumentChunks).where(DocumentChunks.document_id == document_id)
        ).all()

        for chunk in document_chunks:
            session.delete(chunk)
            session.commit()

        if not delete_document_request_body.delete_to_retry:
            session.delete(document)
            session.commit()

        session.close()

        return JSONResponse(
            content={"message": "Document deleted successfully"},
            status_code=status.HTTP_200_OK,
        )


@kb_router.post("/inherit_kb")
async def inherit_knowledge_base(
    inherit_kb_request: InheritKnowledgeBaseRequest,
    db_session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[Users, Depends(get_current_user)],
):
    """
    Inherit knowledge base
    """
    with db_session as session:
        target_kb_id = inherit_kb_request.target_knowledge_base_id
        source_kb_id = inherit_kb_request.source_knowledge_base_id
        if not target_kb_id:
            target_kb = KnowledgeBases(
                name=f"{current_user.username}'s inherited KB",
                description="Inherited from another KB",
                user_id=current_user.users_id,
                is_contextual_rag=True,
            )

            session.add(target_kb)
            session.commit()
            session.refresh(target_kb)
            target_kb_id = target_kb.knowledge_bases_id

        else:
            target_kb = session.exec(
                select(KnowledgeBases).where(
                    KnowledgeBases.knowledge_bases_id == target_kb_id,
                    KnowledgeBases.user_id == current_user.users_id,
                )
            ).first()

            if not target_kb:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Target Knowledge Base not found !",
                )

        if len(target_kb.children) >= 1:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Target Knowledge Base is already inheriting another Knowledge Base",
            )

        source_kb = session.exec(
            select(KnowledgeBases).where(
                KnowledgeBases.knowledge_bases_id == source_kb_id
            )
        ).first()

        if not source_kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source Knowledge Base not found !",
            )

        if target_kb_id in source_kb.parents:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Target Knowledge Base is already inheriting the Source Knowledge Base",
            )

        target_kb_parent = copy.deepcopy(source_kb.parents)
        target_kb_parent.append(source_kb_id)
        target_kb.parents = list(set(target_kb_parent))

        source_kb_children = copy.deepcopy(source_kb.children)
        source_kb_children.append(target_kb_id)
        source_kb.children = list(set(source_kb_children))

        session.add(target_kb)
        session.commit()
        session.add(source_kb)
        session.commit()

        session.close()

        return JSONResponse(
            content={"message": "Knowledge Base inherited successfully"},
            status_code=status.HTTP_200_OK,
        )


@kb_router.delete("/delete_kb/{knowledge_base_id}")
async def delete_knowledge_base(
    knowledge_base_id: str,
    db_session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[Users, Depends(get_current_user)],
    db_manager: Annotated[DatabaseManager, Depends(get_db_manager)],
):
    """
    Delete knowledge base
    """
    with db_session as session:
        query = select(KnowledgeBases).where(
            KnowledgeBases.knowledge_bases_id == knowledge_base_id
        )

        kb = session.exec(query).first()

        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge Base not found !",
            )

        if kb.user_id != current_user.users_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to delete this Knowledge Base",
            )

        # Delete all the documents and its chunks in its old knowledge base
        documents = session.exec(
            select(Documents).where(Documents.knowledge_base_id == knowledge_base_id)
        ).all()
        for doc in documents:
            db_manager.delete_file(
                object_name=doc.file_path_in_minio,
                document_id=doc.documents_id,
                knownledge_base_id=kb.knowledge_bases_id,
                is_contextual_rag=kb.is_contextual_rag,
                delete_to_retry=False,
            )

            document_chunks = session.exec(
                select(DocumentChunks).where(
                    DocumentChunks.document_id == doc.documents_id
                )
            ).all()

            for chunk in document_chunks:
                session.delete(chunk)
                session.commit()

            session.delete(doc)
            session.commit()

        # Delete the assistants associated with the knowledge base
        assistants_ids = session.exec(
            select(Assistants.assistants_id).where(
                Assistants.knowledge_base_id == knowledge_base_id
            )
        ).all()

        for assistant_id in assistants_ids:
            db_manager.delete_assistant(assistant_id=assistant_id)

        parents_copy = copy.deepcopy(kb.parents)
        # Delete the knowledge base from the parents of this knowledge base. Mean that this deleted knowledge base is no longer inheriting its parents
        for parent_id in kb.parents:
            parent_kb = session.exec(
                select(KnowledgeBases).where(
                    KnowledgeBases.knowledge_bases_id == parent_id
                )
            ).first()

            parent_kb_children = copy.deepcopy(parent_kb.children)
            parent_kb_children = [
                child for child in parent_kb_children if child != kb.knowledge_bases_id
            ]
            parent_kb.children = list(set(parent_kb_children))

            session.add(parent_kb)
            session.commit()

        # Delete the knowledge base from the children of its parents. Mean that the knowledge base is no longer inheriting its parents
        queue = deque(kb.children)
        parents_copy.append(kb.knowledge_bases_id)

        while queue:
            child_id = queue.popleft()
            child_kb = session.exec(
                select(KnowledgeBases).where(
                    KnowledgeBases.knowledge_bases_id == child_id
                )
            ).first()

            child_kb_parents = copy.deepcopy(child_kb.parents)
            child_kb_parents = [
                parent for parent in child_kb_parents if parent not in parents_copy
            ]
            child_kb.parents = list(set(child_kb_parents))

            session.add(child_kb)
            session.commit()

            queue.extend(child_kb.children)

        # Finally, delete the knowledge base itself
        session.delete(kb)
        session.commit()
        session.close()

        return JSONResponse(
            content={"message": "Knowledge Base deleted successfully"},
            status_code=status.HTTP_200_OK,
        )
