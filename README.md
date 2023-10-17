# Music genre classification

**Цель исследования**

Разработать модель, позволяющую классифицировать музыкальные произведения по жанрам.

**Задачи:**

- загрузка и ознакомление с данными;
- предварительная обработка;
- EDA;
- разработка новых синтетических признаков,
- проверка на мультиколлинеарность,
- отбор финального набора обучающих признаков,
- выбор и обучение моделей,
- итоговая оценка качества предсказания лучшей модели,
- анализ важности ее признаков.

**Подготовка данных**

Выполнено преобразование типов данных, синтезирован новый признак который использовался в моделе `duration_class`, а также произведено явное преобразование данных. Выявлены явные и неявные дубликаты, выполнен анализ аномальных значений в данных.


**Мультиколлениарность**

Выполнено исследование исходных данных на наличие мультиколлениарности среди имеющихся регрессоров. Найденые регрессоры по средством оценки вздутия дисперсии и значений корреляций Пирсона и Спирмана, которые в последствие не вошли в обучающую выборку (`loudness`, `danceability`, `acousticness`, `valence` `energy` и `duration_ms`)


**Выбор модели**

Для базовых моделей использовались `LogisticRegression` и `RandomForestClassifier`, эти модели выбраны для обзора распределения признаков. Основной моделью была модель градиентного бустинга `CatBoost`, как одна из лучших моделей для решения задачи мультиклассификации. 

**Метрика**

**F (F beta)**

В данной задаче была выбрана метрика F beta, так как она лучше всего подходит для оценки качества модели при решении задачи мультиклассификации.


Формула для метрики F beta ниже

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mi>F</mi>
    <mi>&#x3B2;</mi>
  </msub>
  <mo>=</mo>
  <mo stretchy="false">(</mo>
  <mn>1</mn>
  <mo>+</mo>
  <msup>
    <mi>&#x3B2;</mi>
    <mn>2</mn>
  </msup>
  <mo stretchy="false">)</mo>
  <mfrac>
    <mrow>
      <mtext>precision</mtext>
      <mo>&#xD7;</mo>
      <mtext>recall</mtext>
    </mrow>
    <mrow>
      <msup>
        <mi>&#x3B2;</mi>
        <mn>2</mn>
      </msup>
      <mtext>precision</mtext>
      <mo>+</mo>
      <mtext>recall</mtext>
    </mrow>
  </mfrac>
  <mo>.</mo>
</math>


**Обучение моделей**

Показатели метрик для валидационной выборки:
- `LogisticRegression`
    - F-beta 0.05
- `RandomForestClassifier`
    - F-beta 0.46
- `CatBoostClassifier`
    - F-beta 0.94
 

**Web сервис**
![alt text](https://sun9-55.userapi.com/impg/BZcEdDeIL4j7xHBRdCQAZPNKjATDNUUVVl_U5g/MZSJ83Nmklg.jpg?size=2560x1499&quality=95&sign=bcddfd3dbb099ab34864d2d3ddb1aac0&type=album)

![alt text](https://sun9-78.userapi.com/impg/Essl8bgqUc-B32YUcaQqcE9JeiSJ0Ucrwiqmag/KyQBGPJ-y9Y.jpg?size=2560x1589&quality=95&sign=634466a01981042a0d6dc752ed427d86&type=album)
