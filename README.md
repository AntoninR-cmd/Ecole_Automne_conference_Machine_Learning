# Contexte :

L'objectif de ce projet est de quantifier l'évolution de performance d'un modèle de reconnaissance d'image lorsque les données à prédire sont éloignées des données d'entrainement. 
Ce projet prend ses racines dans l'archéo botanique, un domaine ou l'identification d'espèces de végétaux issues de fouilles peut s'avérer délicat à cause de la déterioration des sujets par les époques. On s'interroge sur la pertinence d'utiliser un modèle de machine Learning pour identifier un tel fossile. Est ce qu'un reseau de neurone convolutif peut permettre d'identifier avec certitude l'espèce d'un fossile ou bien est-ce que l'expertise des archéologues reste la methode la plus sure.
Cette reflexion s'inscrit dans la continuité d'une autre étude réalisé en 2025 par M. Bonhomme et al. qui semblait mettre en avant que l'architecture la plus efficace pour répondre à cette question est l'utilisation de réseau de neurones convolutifs adoptant la structure VGG19.
Nous disposions des mêmes données d'origine que l'article à savoir d'une cohorte de 3 149 images de graines scindées en 3 espèces distinctes. Chaque espèce se sub-divise en 2 statuts : les graines domestiquées et les graines sauvages. A cette cohorte nous avons réalisé un ensemble de prétraitement et un ajout de graines issues de fouilles archéologiques ainsi que d'autres graines contemporaines appartenant à d'autres espèces pour les fins d’évaluation des modeles, pour un total de 6 693 images.

Pour répondre à notre problématique nous avons opté pour une méthodologie progressive ou on a stratifier différents niveau d'éloignement des données de test et de train.

Pour chaque strates, nous posons trois bases d'entrainement comparables pour chaque niveaux : d'abord un modèle entrainé seulement sur les graines domestiquées, puis seulement sur les graines sauvages et enfin sur la combinaison des deux. De cette manière nous pouvons en plus tenter de savoir quels type de graines pourraient permettre une meilleure prédiction et ou que la quantité générale des données fournisse des résultats plus pertinents.
Le premier niveau d'analyse concerne la mise en place d'un modèle simple permettant d'identifier l’espèce d'une graine contemporaine. On s'assure ici que le modèle fonctionne pour des données très proches.
Le deuxième niveau ajoute aux données à prédire des images de graines issues de fouilles. Elles ont étaies détériorés mais sont supposées conserver des caractéristiques communes avec les graines contemporaines.
Enfin dans le dernier niveau on test les limites du modèles et sa capacité à rejeter des nouvelles espèces de graines qui n'ont pas participé à l'élaboration du modèle. Pour cette dernière étape on met en place une nouvelle stratégie, une classe de rejet qui sous un certain seuil de confiance dans la prédiction, rejette l'appartenance de l'image à l'une des espèces à prédire.

# Contenu :

Notre projet se compose donc de [...] fichiers, un dossier "DATA" composé des données issue de l'article sur lequel nous avons basé notre étude ainsi que de données rajoutés. Il y a aussi un dossier "rda", qui contient des fichier ".rda" qui permet de qualifier les images, ainsi que les poids des modèles entrainées précédemment sous format ".h5". C'est grâce au dossier rda que l'on peut distinguer les graines domestiques des graines sauvages par exemple, ainsi que les données utilisées pour l'entrainement des modèles des données de test, il étiquette nos données. 
Un dossier "R" qui contient les différents modèles suivant la hiérarchisation établie précédemment, ainsi que les scripts utilitaires. 
Les scripts "_bis" concernent les modèles entrainés uniquement sur les "domestiquées" ou les "sauvages" et d'autre part les scripts sans ce suffixe, porte sur la combinaison des deux statuts.
Le script "Test4.R" concernent l'evaluation des modèles du niveau 1 avec l'ajout des graines archéologiques à prédire. 
Et enfin les scripts "Test5" mettent en œuvre la classe de rejet. 

# Utilité :

Ce projet s'adresse à tous les archéobotaniques qui souhaitent confronter leur identification avec un modèle de machine Learning qui ne tient compte que de l'aspect des graines modernes sur une large base de données.
Il pourraient également s'adapter facilement à d'autre domaines, comme par exemple la zoologie ou botanique, dès lors qu'il est possible de hiérarchiser la qualité des données à classer.

# Mode d'emploi :

Les différents scripts renvoient tous des graphiques qui permettent d'évaluer l'accuracy et la loss des modèles en fonction des epochs, ainsi que le tableau de contengence final de predictions contre les données à classer. Si l'objectif est de tester un résultat d'un niveau de la méthodologie, il suffit de se rendre dans le script correspondant et d’exécuter le code, en vérifiant que les chemins vers les dossiers sont correctes. 
Si l'objectif est de réitérer les analyses avec des images différentes il est nécessaire d'ajouter un nouveau dossier au nom de l’espèce dans le fichier DATA, utiliser le script de création de fichier ".rda", puis de modifier les paramètres dans le script pour relier les images au modèles.
