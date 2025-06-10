from fastapi import APIRouter 
from sqlalchemy import select 
from app.database import async_session_maker 
from app.src.schemas import UUser
from app.dao.dao import UserDAO
router = APIRouter(prefix='/users', tags=['Страница пользователя'])

@router.get("/", summary="Получить всех пользователей")
async def get_all_users():
    async with async_session_maker() as session: 
        return await UserDAO.find_all()
@router.get("/{id}", summary="Получить одного пользователя по id")
async def get_user_by_id(user_id: int) -> UUser | dict:
    rez = await UserDAO.find_one_or_none_by_id(user_id)
    if rez is None:
        return {'message': f'Пользователь с ID {user_id} не найден!'}
    return rez