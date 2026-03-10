from asyncpg import Connection


class UsersQueries:
    def __init__(self, conn: Connection):
        self.conn = conn

    async def add_tg_user(self, tg_id: int, first_name: str, last_name: str, is_created: bool = True):
        """
        xmax > 0 при успешной вставке
        xmax == 0 при ON CONFLICT -> UPDATE
        """
        query = '''
        INSERT INTO users (tg_id, tg_first_name, tg_last_name, is_created) VALUES ($1, $2, $3, $4)
        ON CONFLICT (tg_id) WHERE tg_id IS NOT NULL
        DO UPDATE SET tg_first_name = excluded.tg_first_name, tg_last_name = excluded.tg_last_name
        RETURNING id, XMAX
                '''
        record = await self.conn.fetchrow(query, tg_id, first_name, last_name, is_created)
        return record.values()


    async def add_web_user(self, email: str, hashed_password: str, name: str, is_created: bool = False):
        query = '''
        INSERT INTO users (email, passw, web_name, is_created) VALUES ($1, $2, $3, $4)
        ON CONFLICT (email) WHERE email IS NOT NULL
        DO NOTHING
        RETURNING id
        '''
        user_id = await self.conn.fetchval(query, email, hashed_password, name, is_created)
        return user_id