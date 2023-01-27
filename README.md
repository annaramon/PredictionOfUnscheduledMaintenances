# PREDICTIVE ANALYTICS USING BIG DATA TECH

L’objectiu d’aquest projecte és predir potencials futures ocurrències en el funcionament
d’un avió per poder arribar a evitar-les. Les dades que se’ns han facilitat per a l’estudi
són un DW i una col·lecció de fitxers CSV. El treballar amb un volum de dades tan gran
sumat a la semi-estructuració de les dades dels CSV fan que haguem de treballar amb un
model ELT i no ETL. El projecte consta de tres data processing pipelines.
La primera pipeline genera una matriu amb els KPIs determinats com a influents en la
possible fallada inesperada de l’activitat d’un avió. Per poder aplicar un algorisme
supervisat necessitem associar un label (manteniment o no) a cada instància que tinguem.
Assumim que tots els avions registrats en la BD AMOS en la taula operationinterruption
han tingut un manteniment (unscheduled maintenance) i filtrem pel sensor d’alerta que
ens interessa, el 3453. El nostre objectiu és predir a 7 dies vista una possible fallada, per
tant, cal deixar constància de l’avió, del dia en què es produeix el manteniment (assumim
que és starttime) i dels 6 dies anteriors. En cas que tinguem informació (tan del CSV: avg
sensor, com del DW: FH, FC, DY) sobre aquell avió en un d’aquests 7 dies li afegim el
label Maintenance, en cas que no en tinguem, li donem un label NonMaintenance. Per
extreure les dades del sensor de cada avió i calcular el average, traiem les dades de tots
els vols que tenim informats i al final de tot agrupem per avió i dia fent l’average del valor
del sensor. D’aquesta manera evitem errors com seria fer la mitjana de les mitjanes
diàries, que no és el que desitgem. L’output d’aquesta pipeline, és a dir, la matriu generada
és guarda en un fitxer (management_matrix) per tal de que no sigui estrictament
necessària executar-la cada vegada, si no que si ja s’ha executat, es pugui carregar.
La segona pipeline entrena el model predictiu, un decision tree, separant les dades en un
70% per training i un 30% de test. Per defecte, al llegir les dades de la matriu
emmagatzemada es defineixen com a strings. Això suposa un problema per a
l’entrenament de l’arbre i per evitar-ho, el primer que fem és canviar l’estructura de les
dades, adaptant-la a integers i doubles i transformant els labels a binari (NonMaintenance
= 0, Maintenance = 1). El següent pas és agregar en un vector les features que determinen
l’absència o no d’un manteniment en un avió. Finalment, entrenem el model i
l’emmagatzemem, com en el cas anterior (analysis_tree). L’accuracy obtingut depèn de
cada iteració, ja que la partició de dades és aleatòria però ronda sobre el 88-93%. A banda
de la precisió, també hem calculat altres mètriques com es pot observar al codi.
L’última pipeline determina si, donat un avió i un dia, tindrà un manteniment inesperat o
no. Suposem que tots els inputs que es donaran estan representats en el DW i en la
col·lecció de CSV, de manera que el primer que fem és imputar les KPIs que hem
determinat que defineixen el l’absència o la presencia de manteniments inesperats (FH,
FC, DY del DW i el average del sensor 3453 dels CSVs). Enviem la tupla al decision tree
entrenat en la pipeline anterior (analysis_tree) i obtenim un output indicant si en els futurs
7 dies es veurà o no afectada l’activitat de l’avió determina.


![pipelines](https://user-images.githubusercontent.com/88190336/215083546-4da7e44b-a82c-4aba-b092-311fc62f0376.png)

