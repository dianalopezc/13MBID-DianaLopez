## Sobre los archivos de este directorio

El archivo incluido de [requerimientos](uv_requirements.txt) está generado desde el gestor UV con el comando:

~~~ bash
uv export --format=requirements.txt >> config/requirements.txt
~~~

No necesariamente va a ser compatible con lo expresado en el archivo general de [entornos](../Entornos.md) para las opciones disponibles. En caso de no utilizar UV se recomienda seguir los pasos para generar el archivo requirements.txt vía el comando [make](../Makefile) o el script [.bat](../crear_archivo_requerimientos.bat) según corresponda.