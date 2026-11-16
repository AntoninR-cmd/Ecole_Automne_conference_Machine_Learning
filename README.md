# Contexte

L'objectif de ce projet est de quantifier l'évolution des performances d'un modèle de reconnaissance d'images lorsque les données à prédire s'éloignent des données d'entraînement.

Ce projet puise ses racines dans l'archéobotanique, un domaine où l'identification d'espèces végétales issues de fouilles peut s'avérer délicate en raison de la détérioration des spécimens au fil du temps. Nous nous interrogeons sur la pertinence d'utiliser un modèle de *machine learning* pour identifier de tels fossiles. Dans quelle mesure un réseau de neurones convolutif peut-il identifier correctement l'espèce d'un fossile à partir de son apparence, malgré les altérations dues au temps ?

Cette réflexion s'inscrit dans la continuité d'une autre étude réalisée en 2025 par M. Bonhomme et al., qui semble mettre en avant l'utilisation de réseaux de neurones convolutifs adoptant l'architecture VGG19 comme approche particulièrement efficace pour répondre à cette problématique.

Nous disposions des mêmes données d'origine que celles utilisées dans l'article, à savoir une cohorte de 3 149 images de graines réparties en trois espèces distinctes. Chaque espèce se subdivise en deux statuts : les graines domestiquées et les graines sauvages. À cette cohorte, nous avons appliqué un ensemble de prétraitements et ajouté des graines issues de fouilles archéologiques, ainsi que d'autres graines contemporaines appartenant à des espèces différentes, utilisées pour l'évaluation des modèles. L'ensemble représente un total de 6 693 images.

Pour répondre à notre problématique, nous avons opté pour une méthodologie progressive, dans laquelle nous avons stratifié différents niveaux d'éloignement entre les données d'entraînement et les données de test.

Pour chaque strate, nous avons défini trois bases d'entraînement comparables : un modèle entraîné uniquement sur les graines domestiquées, un deuxième uniquement sur les graines sauvages et un troisième sur la combinaison des deux. De cette manière, nous pouvons également chercher à déterminer quel type de graines permet d'obtenir les meilleures prédictions, et si l'augmentation de la quantité globale de données permet d'obtenir des résultats plus pertinents.

Le premier niveau d'analyse concerne la mise en place d'un modèle simple permettant d'identifier l'espèce d'une graine contemporaine. Nous nous assurons ici que le modèle fonctionne sur des données très proches de celles utilisées pour son entraînement.

Le deuxième niveau ajoute aux données à prédire des images de graines issues de fouilles archéologiques. Ces dernières sont détériorées, mais sont supposées conserver des caractéristiques communes avec les graines contemporaines.

Enfin, dans le dernier niveau, nous testons les limites du modèle et sa capacité à rejeter de nouvelles espèces de graines qui n'ont pas participé à son entraînement. Pour cette dernière étape, nous mettons en place une nouvelle stratégie : une classe de rejet qui, lorsque le niveau de confiance de la prédiction est inférieur à un certain seuil, rejette l'appartenance de l'image à l'une des espèces connues du modèle.

# Contenu

Notre projet se compose donc de [...] fichiers et dossiers. Un dossier « DATA » regroupe les données issues de l'article sur lequel nous avons basé notre étude, ainsi que les données ajoutées.

Il y a également un dossier « rda », qui contient des fichiers au format « .rda » permettant de qualifier les images, ainsi que les poids des modèles entraînés précédemment, au format « .h5 ». C'est grâce aux informations contenues dans les fichiers RDA que l'on peut distinguer, par exemple, les graines domestiquées des graines sauvages, mais aussi différencier les données utilisées pour l'entraînement des modèles de celles utilisées pour les tests. Ces fichiers permettent donc d'étiqueter et de caractériser nos données.

Un dossier « R » contient les différents modèles, organisés suivant la hiérarchisation établie précédemment, ainsi que les scripts utilitaires.

Les scripts portant le suffixe « _bis » concernent les modèles entraînés uniquement sur les graines « domestiquées » ou « sauvages ». À l'inverse, les scripts sans ce suffixe portent sur la combinaison des deux statuts.

Le script « Test4.R » concerne l'évaluation des modèles du niveau 1, avec l'ajout de graines archéologiques parmi les données à prédire.

Enfin, les scripts « Test5 » mettent en œuvre la classe de rejet.

# Utilité

Ce projet s'adresse aux archéobotanistes qui souhaitent confronter leurs identifications à celles d'un modèle de *machine learning* entraîné uniquement à partir de l'aspect de graines modernes, issues d'une large base de données.

Il pourrait également être adapté à d'autres domaines, comme la zoologie ou la botanique, dès lors qu'il est possible de hiérarchiser les niveaux de qualité des données à classer.

# Mode d'emploi

Les différents scripts produisent des graphiques permettant d'évaluer l'accuracy et la loss des modèles en fonction du nombre d'époques (*epochs*), ainsi qu'une matrice de confusion finale comparant les prédictions aux classes réelles des données à classer.

Si l'objectif est de tester un résultat correspondant à un niveau de la méthodologie, il suffit de se rendre dans le script correspondant et d'exécuter le code, en vérifiant au préalable que les chemins d'accès aux dossiers sont corrects.

Si l'objectif est de réitérer les analyses avec des images différentes, il est nécessaire d'ajouter un nouveau dossier portant le nom de l'espèce dans le dossier « DATA », d'utiliser le script de création des fichiers « .rda », puis de modifier les paramètres dans les scripts afin de relier les nouvelles images aux modèles.
