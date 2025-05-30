# Предсказание динамики физической системы с помощью нейросетей (годовой проект)
Куратор - Марк Блуменау (TG: https://t.me/markblumenau, GitHub: https://github.com/markblumenau).

Команда:
- Михаил Мокроносов (TG: https://t.me/idodir, GitHub: https://github.com/iDodir);
- Владислав Семенов (TG: https://t.me/Vladislav_iSemenov, GitHub: https://github.com/Vladislav-IS);
- Матвей Спиридонов (TG: https://t.me/spiridonovms, GitHub: https://github.com/matveyspiridonov).

Цель проекта - провести многоклассовую классификацию событий (частиц) в эксперименте LHCb (детектор на адронном коллайдере). Будут опробованы как classic ML, так и DL подходы.

## Чекпоинт 4. Сервис
Весь код к этому чекпоинту реализован в папке `service`. Папка `telegram` к чекпоинту не относится, телеграм-бот будет реализован в перспективе.

### Структура сервиса
Структура серверной части (папка `fastapi`):
- `main.py` – точка входа программы;
- `router.py` – файл API-роутера, в котором определены функции обработ-ки запросов;
- `settings.py` – файл, в котором определена конфигурация программы;
- `services.py` – файл, содержащий функции для обучения моделей;
- `log_config.json` – файл конфигурации логирования для сервера uvicorn;
- `content` – папка, содержащая готовые файлы для отправки клиенту (в частности, файл с выполненными разведочным анализом данных eda.pdf);
- `requirements.txt` и `Dockerfile`;
- `logs` – папка с логами;
- `models` – папка с обученными моделями.
Папки `logs` и `models` создаются автоматически, если на момент запуска сервера их не существовало.

Структура клиентской части(папка `streamlit`) :
- `1_💻_Intro.py` – файл стартовой страницы приложения;
- `pages/ 2_📊_EDA.py` – файл страницы просмотра разведочного анализа данных;
- `pages/ 3_🤖_Classic_ML.py` – файл страницы обучения и инференса моделей;
- `client_funcs.py` – файл с функциями запросов к серверу, а также с неко-торыми функциями, которые не связаны с клиент-серверным взаимодействием, но используются на всех страницах;
- `settings.py` – файл, в котором определена конфигурация программы;
- `requirements.txt` и `Dockerfile`.
Эмозди в названии файлов страниц отображаются в приложении в качестве иконок.

### Развертывание сервера и клиента
#### Локльное развертывание

После скачивания исходников проекта из репозитория и установки зависимостей сервер можно запустить из консоли следующей командой:

`uvicorn main:app --host 0.0.0.0 --port 8000 --log-config log_config.json --reload`

В свою очередь, Streamlit-клиент запускается по команде:

`streamlit run 1_💻_Intro.py`

Важно: переменная `FASTAPI_URL` в `service\streamlit\settings.py` должна быть равна `http://127.0.0.1:8000/`.

#### Развертывание в Docker

В папке service находится файл docker-compose.yml. Чтобы запустить Docker-контейнеры, необходимо выполнить команду:

`docker-compose -f docker-compose.yml up`

Важно: переменная `FASTAPI_URL` в `service\streamlit\settings.py` должна быть равна `http://fastapi:8000/`.

#### Развертывание на VPS

В рамках чекпоинта удалось развернуть сервер и клиент на render.com. Для этого понадобилось по шаблону создать 2 репозитория:
- https://github.com/Vladislav-IS/ai-year-project-service-fastapi - сервер;
- https://github.com/Vladislav-IS/ai-year-project-service-streamlit - клиент.
Данные репозитории были загружены на сайт render.com и развернуты в качестве веб-серсвисов.

Из недостатков выбранной платформы VPS следует отметить нестабильность работы развернутых сервисов. Также пришлось избавиться от мультипроцессорности при обучении моделей, поскольку продоствленные мощности не позволили адекватно ее реализовать.
FastAPI-сервер доступен по ссылке: https://ai-year-project-service-qeke.onrender.com. В свою очередь, Streamlit-клиент доступен по ссылке: https://ai-year-project-service-streamlit.onrender.com.

#### Интерфейс Streamlit
Интерфейс Stremlit содержит следующие страницы:
- начальная страница "Intro";
- страница просмотра разведочного анализа данных "EDA";
- страница обучения и инференса моделей "Classc ML".

Страница "Intro" созержит информацию о проекте и ссылку на репозиторий:
![image](https://github.com/user-attachments/assets/7a489245-06d3-4962-a267-ff8aad1368ea)

Страница "EDA" содержит следующие кнопки: 
- "Скачать PDF" - позволяет загрузить PDF-файл с проведенным анализом данных;
- "EDA в реальном времени" - позволяет выполнить анализ загруженных вручную данных в режиме реального времени.

Пример работы со страницей "EDA": [видео](https://github.com/user-attachments/assets/a0cd9b70-959c-4ebf-add4-c304fd802b2a)

В свою очередь, страница "Classic ML" содержит кнопки обучения моделей, выполнения предсказаний и работы со списком моделей. 

Обучение моделей выполняется после загрузки файла с датасетом, ввода ID моделей и гиперпараметров. Пример обучения моделей: [видео](https://github.com/user-attachments/assets/9f7691ca-de74-43d7-b0ec-be61d4949f5d).

Также на странице "Classic ML" доступен просмотр и модификация списка обученных моделей:
- установка текущей модели для инференса
- снятие модели с инференса;
- удаление модели по ID;
- удаление всех моделей;
- сравнение качества моделей по метрикам.

Пример установки текущей модели для инференса: [видео](https://github.com/user-attachments/assets/1bd2413f-d2fe-484a-af67-b332f3f64941).

Пример удаления модели из списка: [видео](https://github.com/user-attachments/assets/b671b00c-ba4a-4e0c-b0fc-a74e6f188e97).

Пример сравнения качества моделей: [видео](https://github.com/user-attachments/assets/921e8d26-fa85-40b5-9a60-caa71c5f560e).

Предсказания также выполняются после загрузки файла с данными (перед этим необходимо установить какую-либо модель для инференса). Пример выполнения предсказаний: [видео](https://github.com/user-attachments/assets/2fc9a318-515d-494f-be87-dc0c13a64cef).
