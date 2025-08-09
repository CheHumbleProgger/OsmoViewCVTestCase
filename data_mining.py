import pandas as pd
import os
import requests
from urllib.parse import urlparse


def download_image(url, save_dir='downloaded_images'):
    """
    Скачать изображение по данному URL и сохранить в данную директрию.

    Args:
        url (str): URL изображения
        save_dir (str): директория для сохранения

    Returns:
        str: путь к сохраненному изображению или None при ошибке
    """
    try:
        # Создаем директорию, если ее еще нет
        os.makedirs(save_dir, exist_ok=True)

        # Берем ссылку и предобрабатываем ее
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:  # Если ссылка не ведет к такому файл напрямую, то попробуем так:
            filename = f"image_{hash(url)}.jpg"

        # Возбмем картинку и сохраним ее
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        save_path = os.path.join(save_dir, filename)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        return save_path

    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")
        return None


def get_data_from_df(dataframe_name: str, filename: str, path_to_data: str) -> None:

    """
    Функция для загрузки данных с помощью ДатаФрейма со ссылками на них.
    Args:
        dataframe_name: имя датафрейма. Предполагается, что он лежит в директории './data' и имеет расширение .csv
        filename: имя файла, описывающего данные, будет далее использоваться для меток класса
        path_to_data: директория, куда будут сохраняться данные
    Returns:
        None, но создает директорию, где лежат данные
    """
    df_path = os.path.join('./data', dataframe_name+str('.csv'))
    df = pd.read_csv(df_path, names=[filename])
    list_of_links = df[filename].to_list()
    path_to_data = os.path.join(path_to_data, filename)
    os.makedirs(path_to_data, exist_ok=True)

    for i in range(len(list_of_links)):
        download_image(list_of_links[i], path_to_data)
