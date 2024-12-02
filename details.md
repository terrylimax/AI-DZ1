В рамках данной работы сделано следующее: 
- Как датасет выглядел в начале работы
<img width="1351" alt="Screenshot 2024-11-27 at 22 32 18" src="https://github.com/user-attachments/assets/a2cd286f-61d9-471b-9469-fa1b3a3ea105">
- Построил дашборд в ydata_profiling, html для скачивания доступен по ссылке: https://github.com/terrylimax/AI-DZ1/blob/main/profiling_report.html
  
  ![image](https://github.com/user-attachments/assets/48a33489-59fb-420f-bed0-44ae6c4729ed)
- Перед тем как избавиться от пустых значений, обработал столбцы torque, mileage', 'engine', 'max_power'  
    - torque - с помощью регулярного выражения преобразовал в столбцы nm(Ньютон-метр, единица измерения крутящего момента), rpm(обороты в минуту), kgm(устаревшая единица измерения крутящего момента)
      
  ![Screenshot 2024-11-27 at 22 40 36](https://github.com/user-attachments/assets/653bc0a5-2e06-4f72-a070-cd4a294336ce)
  
  ![image](https://github.com/user-attachments/assets/ac18b4ec-eb07-47bd-8011-12f583ed82e7)
    - отфильтровал mileage, engine, max_power
      
  ![image](https://github.com/user-attachments/assets/097f0a41-398d-485f-be81-46e3320d7c3b)
-  заполнили пустые ячейки медианой с помощью SimpleImputer

![Screenshot 2024-11-27 at 23 28 45](https://github.com/user-attachments/assets/1201a2df-ddc8-49e2-adfd-d9f37001d928)
- применяем то же самое для тестового набора данных

![Screenshot 2024-11-27 at 23 32 19](https://github.com/user-attachments/assets/48aedb31-7046-4b2f-94a6-ed9b77fb7c33)

- Удалил дубликаты, преобразовал engine и seats в числовой тип, вывел описательные статистики для числовых и категориальных признаков
- Построил графики взаимодействия признаков с целевой перменной и сежду собой. У некоторых признаков есть зависимость - чем больше год, тем более вероятно будет дороже цена, обратная ситуация с пробегом (ниже пробег, выше цена), в mileage не вижу явной зависимости, engine, max_power, nm есть зависимость, у seats нет зависимости с price, у rpm не вижу явной зависимости. engine/max_power коррелируют между собой, также engine/nm, max_power/nm
  
![image](https://github.com/user-attachments/assets/f3c1d6b2-9c4c-445f-a5cb-07fbeb163c04)
- матрица Корреляций

![image](https://github.com/user-attachments/assets/2b187eed-d407-4873-b946-6ecb65e6e40c)
-также, нет графиков, которые бы брали во внимание категории - вполне вероятно, что машина, которая была у 5 владельца стоит дешевле чем у первого или второго, и механическая коробка дешевле автомата - можем отобразить это на boxplot - вывел описательные статистики - на скрине ниже видно, что механическая коробка в целом дешевле автомата

![image](https://github.com/user-attachments/assets/b53e7566-f6cf-4ab7-87fd-dc97b782835a)
- Далее сделал бейзлайн модели линейной регрессии
- Стандартизировал значения в тренировочном и тестовом датасете с помощью StandardScaler, MSE R^2 не изменились
- После регуляризации Lasso качество ухудшилось
- Прошелся через GridSearch по набору моделей Lasso, ElasticNet, MSE R^2 не улучшились
- Применил One Hot Encoding к категориальным признакам, перебрал Ridge модели с помощью GridSearch, MSE R^2 заметно улучшились
- Создал бизнесовую кастомную метрику с долей объектов, для которых спрогнозированная цена в пределах 10% от реальной - по кол-ву процентов Lasso выигрывает, но я бы протестировал еще на других датасетах

- скрины работы сервиса (файл [сервиса](https://github.com/terrylimax/AI-DZ1/blob/main/fastapi_app.py))
![Screenshot 2024-12-03 at 00 17 43](https://github.com/user-attachments/assets/5ee6b9f0-754d-4126-b49a-47152252174e)
1. на вход в формате json подаются признаки одного объекта, на выходе сервис выдает предсказанную стоимость машины
![Screenshot 2024-12-03 at 00 18 15](https://github.com/user-attachments/assets/544ba9bb-06df-47eb-acc0-9f4c1f2d7390)
2. на вход подается csv-файл с признаками тестовых объектов, на выходе получаем файл с +1 столбцом - предсказаниями на этих объектах
Загружаем X_test
![Screenshot 2024-12-03 at 00 19 24](https://github.com/user-attachments/assets/7657093c-2d4a-42a1-a7d8-ea1b76163ef5)
Выгружаем результат
![Screenshot 2024-12-03 at 00 20 20](https://github.com/user-attachments/assets/5fdcfe28-49fc-4659-814f-8264a69bc72e)
Смотрим результат в Jupyter
![Screenshot 2024-12-03 at 00 22 47](https://github.com/user-attachments/assets/5413fd7d-31e3-47aa-b355-ea1352167b52)

Что не смог

Не смог обработать names - поначалу сделал последовательное кодирование 1..x, но потом вспомнил, что это скорее во вред, и на степике рассказывали про счетчики со сглаживанием - времени не хавтило доделать
