from asyncpg import Connection

from web.utils.anything import ImgStatuses
from web.utils.logger_config import log_event


class ImagesQueries:
    def __init__(self, conn: Connection):
        self.conn = conn

    async def save_tg_files_meta(self, tg_id: int, file_ids: list):
        query = '''
        WITH user_id_from_tg AS (
            SELECT id FROM users WHERE tg_id = $1
        )
        INSERT INTO img_appeals AS ia (user_id, file_id)
        SELECT
            (SELECT id  FROM user_id_from_tg) AS user_id,
            (SELECT file_id FROM UNNEST($2::text[]) AS file_id) 
        RETURNING ia.id
        '''
        res = await self.conn.fetch(query, tg_id, file_ids)
        return res


    async def update_images_status(self, ml_resp_img_list: list[dict]):
        """
        ml_resp_img_list: Ожидает следующее в списке. Нечувствителен к ключам словаря.
        Только к типам данных и порядку пар "ключ-значение":)

        [
            {
                'img_id': 1,
                'cloud_archive_path': f'ocr/1.tar.gz', # предварительный путь к архиву в S3
                'text': '123',
                'word_count': 1
            },
        ]
        """
        if not ml_resp_img_list:
            log_event('Нет фото для обновления статуса', level='CRITICAL')
            return []

        img_ids, archive_cloud_paths, text_results, word_counts = zip(*map(lambda x: x.values(), ml_resp_img_list))
        query = '''
        UPDATE img_appeals ia
        SET cloud_archive_path = pd.archive_cloud_paths, 
            words_count = pd.words_counts, 
            inference_text = pd.text_results,
            status_id = $5
        FROM (
            SELECT img_ids, archive_cloud_paths, words_counts, text_results
            FROM UNNEST($1::int[], $2::text[], $3::int[], $4::text[]) 
            AS t(img_ids, archive_cloud_paths, words_counts, text_results) 
        ) AS pd   
        WHERE ia.id = pd.img_ids
        RETURNING ia.id
        '''
        upd_count = await self.conn.fetch(query, img_ids, archive_cloud_paths, word_counts, text_results, ImgStatuses.success)
        return upd_count


    async def rate_ocr_res(self, img_id: int, rate: int):
        query = 'UPDATE img_appeals SET user_rate = $2 WHERE id = $1 RETURNING id'
        res = await self.conn.fetchval(query, img_id, rate)
        return res