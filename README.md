Los datos son los mismos que voy a utilizar en el Trabajo Final de la carrera

LINK a la competencia en Kaggle

https://www.kaggle.com/c/m5-forecasting-accuracy

Se trata de información histórica de ventas de 3049 productos repartidos en 3 categorías (Comidas, Hobbies y Productos del Hogar) y 7 departamentos para 10 tiendas localizadas en 3 estados de Estados Unidos: California, Wisconsin y Texas. Las series de tiempo comprenden desde el 29 de enero de 2011 hasta el 19 de junio de 2016.

El objetivo es encontrar el modelo que mejor prediga las ventas de cada tienda por día durante un mes.

En este caso utilizo el algortimo LightGBM y defino sus hiperparametros a través de una optimización bayesiana.

Utilicé varios snippets para reducir el uso de la memoria (ya que es un dataset grande de 62 millones de filas)

El dataset consiste de los siguientes 3 archivos:

Archivo 1: “calendar.csv” 
Contiene informacion sobre las fechas en las que los productos fueron vendidos.

•	date: fecha en formato a “y-m-d”.
•	wm_yr_wk: EL id del dia de la semana a la que pertenece.
•	weekday: Tipo de dia (Sabado, Domingo, …, Viernes).
•	wday: Id del dia de la semana, empezando por el sábado.
•	month: Mes de la fecha.
•	year: Año de la fecha.
•	event_name_1: Si la fecha incluye un evento, el nombre del evento.
•	event_type_1: Si la fecha incluye un evento, el tipo de evento.
•	event_name_2: Si la fecha incluye un segundo evento, el nombre del evento.
•	event_type_2: Si la fecha incluye un segundo evento, el tipo de evento.
•	snap_CA, snap_TX, and snap_WI: Una variable binaria indicando si las tiendas de CA TX or WI permiten compras SNAP. 1 indica que estan permitidas.

Archivo 2: “sell_prices.csv”
Contiene informacion sobre el precio de los productos vendidos por tienda y fecha.
•	store_id: El id de la tienda donde el producto es vendido. 
•	item_id: El id del producto.
•	wm_yr_wk: El id de la semana.
•	sell_price: El precio del producto para una semana tienda

Archivo 3: “sales_train.csv” 
Contiene la data diaria historica de las ventas en unidades por producto y tienda.

•	item_id: El id del producto.
•	dept_id: E id del departamento al que pertenece el producto.
•	cat_id: El id de la categoria a la que pertence.
•	store_id: EL id de la tienda donde el producto es vendido.
•	state_id: El estado donde la tienda se encuentra localizada.
•	d_1, d_2, …, d_i, … d_1941: El numero de unidades vendida en el dia i empezando por 2011-01-29. 
