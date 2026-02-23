# Dataset: Bank Marketing (con contexto social/económico)

## Descripción general

Este dataset describe los resultados de campañas de marketing telefónicas realizadas por un banco en Portugal.
El objetivo de las campañas era ofrecer a los clientes la apertura de un depósito a plazo.
La variable objetivo (`y`) indica si el cliente aceptó (`yes`) o no (`no`) la oferta.

Fuente de datos:
[UCI Machine Learning Repository – Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

## Referencia académica

Este dataset es de acceso público para investigación.
Detalles en:
S. Moro, P. Cortez y P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems, Elsevier, 62:22-31, junio 2014.
DOI: [10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001)

### Autores

* Sérgio Moro (ISCTE-IUL)
* Paulo Cortez (Universidade do Minho)
* Paulo Rita (ISCTE-IUL)pip install dvc
  Año: 2014

## Información relevante

* Basado en el dataset *Bank Marketing* del repositorio UCI.
* Se añadieron 5 atributos sociales y económicos publicados por el **Banco de Portugal**.
* Estos atributos mejoran significativamente la predicción del éxito de la campaña.
* La tarea de clasificación binaria consiste en **predecir si un cliente contratará un depósito a plazo**.

### Tamaño

* Número de instancias: **41.188**
* Número de atributos: **20 variables de entrada + 1 variable objetivo**

## Variables

### Datos del cliente bancario

1. **age**: edad (numérica)
2. **job**: tipo de trabajo (categórica: *admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown*)
3. **marital**: estado civil (categórica: *divorced, married, single, unknown*)

   * Nota: *divorced* incluye viudos/as.
4. **education**: nivel educativo (categórica: *basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown*)
5. **default**: ¿tiene créditos en default? (*yes, no, unknown*)
6. **housing**: ¿tiene préstamo hipotecario? (*yes, no, unknown*)
7. **loan**: ¿tiene préstamo personal? (*yes, no, unknown*)

### Datos de la última campaña

8. **contact**: tipo de contacto (*cellular, telephone*)
9. **month**: mes del último contacto (*jan, feb, mar, …, nov, dec*)
10. **day_of_week**: día de la semana (*mon, tue, wed, thu, fri*)
11. **duration**: duración del último contacto en segundos (numérica)

* Nota: este atributo influye fuertemente en el resultado.
* No debe usarse para modelos predictivos realistas, ya que solo se conoce después de la llamada.

### Otros atributos

12. **campaign**: número de contactos en esta campaña (numérica)
13. **pdays**: días transcurridos desde el último contacto en campañas anteriores (numérica; 999 = nunca contactado)
14. **previous**: número de contactos previos a esta campaña (numérica)
15. **poutcome**: resultado de la campaña anterior (*failure, nonexistent, success*)

### Contexto social y económico

16. **emp.var.rate**: tasa de variación del empleo – indicador trimestral (numérica)
17. **cons.price.idx**: índice de precios al consumidor – indicador mensual (numérica)
18. **cons.conf.idx**: índice de confianza del consumidor – indicador mensual (numérica)
19. **euribor3m**: tasa Euribor a 3 meses – indicador diario (numérica)
20. **nr.employed**: número de empleados – indicador trimestral (numérica)

### Variable objetivo

21. **y**: ¿el cliente contrató un depósito a plazo? (*yes, no*)

## Valores faltantes

Algunas variables categóricas presentan valores faltantes representados como `"unknown"`.
Se pueden tratar como una categoría propia o imputarse según técnicas de preprocesamiento.

---
