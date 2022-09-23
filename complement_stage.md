# Première partie

1. Un parseur est un analyseur synthaxique. Il permet d'analyser un langage, d'extraire des éléments et les interprete. Ici, notre but est d'utiliser l'outil via un terminal en tapant la commande exécutant l'outil à travers un parseur de la ligne de commande. Nous trouvons que ceci est beaucoup plus simple à utiliser qu'un dictionnaire, car il permet d'avoir un contrôle sur les variables utilisées par le script sans avoir à lire le code pour trouver dans quel(s) module(s) les variables sont créées et appelées. Nous avons utilisé la bibliothèque **argparse** car elle est la plus récente et a une très bonne documentation, avec une bonne gestion des exceptions.

2. Le script principal se sépare en plusieurs modules principaux :

    - ***main.py*** : Exécute le script principal. Il est composé de deux fonctions :
        - *init_main()* qui permet de récupérer les variables à partir du parseur et de tester si les variables d'entrées sont valides et renvoie la fonction *main()*
        - *main()* qui identifie le type de design (par ligation ou par Polymerase Chain Assembly) et lance le design des oligonucléotides. La fonction est récursive dans le cas où design a échoué (car les oligonucleotides obtenus ne respectent pas les conditions établies par l'utilisateur), en modifiant la taille des chevauchements d'oligonucléotides de départ.
    - ***utils.py*** : Contient plusieurs fonctions appelées dans le main.py :
        - *get_args()* contient le parseur et les différentes variables utilisées par l'outil. Il renvoie un namespace contenant toutes les variables.
        - *check_pca/lig_options()* vérifie si les variables appelées sont valides, en fonction du design souhaité.  
        - *setup_pca/lig()* crée un dictionnaire contenant des paramètres supplémentaires à partir des variables d'entrée.
        - *print_options()* renvoie les variables et paramètres utilisés par l'outil.
    - ***design.py*** : Créé les différents blocs et chevauchements d'oligonucléotides à partir des arguments donnés en entrée:
        - *create_design_blocks()* crée les blocs utilisés à partir des fonctions de split.py
        - *design_oligo()* va itérer chaque bloc pour créer leur set de chevauchements.
    - ***compute_tm.py*** : Répété X fois, ce module va calculer la température de fusion pour chaque chevauchement d'oligonucléotides composant un bloc, les classer selon celle-ci et modifier les chevauchements selon le classement :
        - *compute_tm()* va itérer X fois le classement des chevauchements, en appelant des fonctions secondaires permettant de calculer la Tm, classer les chevauchements et modifier leurs séquences. A la fin de l'itération, elle renvoie la fonction **output.output_final_oligo()**
        - *get_tm()* calcule la température de fusion de chaque chevauchement
        - *ranking()* classe les chevauchements en fonction de leur Tm
        - *oligoMod()* modifie la taille des deux chevauchements avec les températures les plus extrêmes et renvoie les nouvelles tailles des chevauchements
        - *get_overlaps()* créé les nouveaux chevauchements à partir de leur nouvelle taille à partir de la fonction **split.make_ligation/pca_overlaps()**

    - ***output.py*** : Renvoi les oligonucléotides modifiés sous format .fasta et .tsv, avec les fichiers d’annotations pour l’indexation et vérification des séquences:
        - *get_threshold()* renvoie les chevauchements avec le meilleur delta Tm à l'itération *i*.
        - *get_bed_dict()* renvoie un dictionnaire contenant des informations d'annotations pour Geneious Prime
        - *get_(pca)_final_output()* renvoie une table contenant les oligonucléotides et les variables utilisées par l'imprimante

    Ces modules principaux sont associés à des modules secondaires permettant d’améliorer la lisibilité du programme
    
    - ***split.py*** : Contient les fonctions permettant de séparer les séquences en fonction de leur état d’entrée (gène complet, bloc, . . .)
        - *make_block()* renvoie un dictionnaire contenant les blocs et leurs caractéristiques
        - *make_subblock_overlaps()* renvoie un dictionnaire contenant les chevauchements de blocs et leurs caractéristiques
        - *make_ligation/pca_overlaps()* renvoie un dictionnaire contenant les chevauchements d'oligonucleotides avant/après modification de leurs tailles
        - *make_(pca)_final_oligos* renvoie un dictionnaire contenant les oligonucléotides finaux formés avec leurs chevauchements finaux
    - ***writers.py*** : Contient les fonctions permettant d’écrire les fichiers de sorties sous différentsformats (.fasta, .tsv, .bed, . . .)
        - *output_final_oligo()* crée à partir des modules **output** et **split** les oligonucléotides et les renvoies sous forme de fichier .tsv/.fasta
        - *write_outputs()* contient toutes les fonctions permettant de renvoyer les différentes données sous forme .tsv, .fasta, .bed


   L'outil fonctionne comme suit :  
      1. La fonction **init_main()** est appelée pour déclarer les variables et renvoie la fonction **main()**.  
      2. La fonction **main()** est appelée pour créer les paramètres spécifiques aux variables déclarées et lance les fonctions **create_design_blocks()** et **design_oligo()**. Si cette dernière renvoie "False", la fonction se rappelle pour relancer un design avec des tailles de chevauchements de départ différents.  
      3. La fonction **design_oligo()** récupère les blocs créés à partir du module ***split*** et de la fonction **create_design_blocks()** et lance **compute_tm()** pour chaque bloc.  
      4. **compute_tm()** calcule la Tm, classe et modifie chaque set de chevauchements d'oligonucléotides un nombre X de fois et renvoie ensuite la fonction **output_final_oligo()** si les chevauchements respectent les conditions de l'utilisateur, sinon renvoie "False".  
      5. **output_final_oligo()** récupère les données de l'itération *i* avec le meilleur delta Tm, crée les oligonucléotides à partir des chevauchements et écrit les oligonucléotides dans des fichiers .tsv/.fasta avec les annotations en .bed.  

      Vous trouverez en pièce jointe un diagramme interactif d'appel des fonctions citées.

      Cette implémentation n'est pas la plus optimale, pour avoir un code plus lisible et fonctionnel, il faudrait transformer les oligonucléotides, les chevauchements et les blocs sous forme de classes (par exemple une classe Oligo(), une classe Block() et une classe Overlaps()). Néanmoins, le fait d'utiliser des modules nous permet de morceler un code large et compliqué en petits modules plus facile à manager. Cela nous permet aussi de récupérer et d'appeler les fonctions depuis un seul fichier, ce qui facilite la réutilisation, le management et le test de l'outil. 



3.  Nous l'avons vu en cours, les tests sont primordiaux pour suivre l'intégration continuel d'un outil et son déployement. Ils nous permettent de nous assurer que les changements effectués dans la base du code n'affecte pas les logiques centrales de celui-ci. Les tests sont écris pour cibler des morceaux spécifiques du code (généralement des fonctions) pour vérifier le bon fonctionnement de ceux-ci, ainsi que le management des exceptions. 
Actuellement, les tests implémentés permettent de vérifier quelques critères:
    - On souhaite vérifier si les différents fonctions de l'outil sont bien conforme à ce qu'on attend d'elles, plusieurs tests sont utilisés pour vérifier ça
    - On souhaite savoir si les blocs ou les oligonucléotides formés à l'aide de l'outil reproduisent bien exactement le gène de référence. Plusieurs tests sont utilisés pour mapper les oligonucléotides/blocs au gène de référence et regarde le taux de couverture. Si le gène n'est pas entièrement couvert, le test est faux.
    - On souhaite tester les différents paramètres automatiquement, plusieurs tests ont étés en utilisant une grille de paramètres pour voir si l'outil ne se termine pas prématurément.  

    Prenons par exemple l'un des tests, **test_oligo_with_gene()**, qui vérifie si les oligonucléotides d'un bloc mappent bien le bloc en question avec une couverture de 100%. Comment le test fonctionne : il appelle l'outil pour créer des oligonucléotides harmonisés à partir d'une séquence d'entrée appelée *test*. On itère à chaque bloc et on vérifie si les oligos reproduisent bien un bloc en additionnant chaque séquence d'oligonucléotide et en comparant la séquence obtenue avec le bloc en question. Si la séquence obtenue n'est pas égal au bloc, le test échoue. 


# Deuxième partie

1. La méthode n'est pas vraiment une méthode dynamique, je pense avoir confondu avec une autre méthode assez similaire, mais qui d'après eux repose sur l'algorithme de Viterbi qui utilise de la programmation dynamique (https://www.frontiersin.org/articles/10.3389/fgene.2022.836108/full). La méthode actuelle consiste en l'exploration d'une table sous forme d'un arbre binaire, à la recherche d'une solution selon la fonction d'exploration de l'arbre. L'algorithme se repose selon trois matrices:

      - Soit *S* la séquence d'entrée de longueur *L*. Le but est de morceler la séquence d'entrée en plusieurs oligonucléotides. Nous souhaitons que les oligonucléotides aient une taille entre *l-4* et *l+4* avec *l* une longueur fixée par l'utilisateur. La première matrice, appelée matrice d'oligonucléotides, va recenser la séquence de tous les oligonucléotides possibles à chaque position de nucléotides de la séquence d'entrée.
      Exemple : prenons la séquence **CATCACCGCGATAGGCTGACAAAGGTTTAA**. La première position est un nucléotide **C**. Les oligonucléotides possibles ont une taille comprises entre *l-4* et *l+4* et on fixe *l* à 15 nucléotides. Ainsi, le premier oligonucléotide possible à la première position **C** aura une longueur de *l-4=11* nucléotides (**CATCACCGCGA**) et le dernier oligonucléotide possible à la position **C** aura une longueur de *l+4=19* nucléotides (**CATCACCGCGATAGGCTGA**).
      La séquence d'entrée va être parcourue position par position, jusqu'à atteindre la position égale à *L-l-4*, car au delà de cette position, il ne sera plus possible d'avoir des oligonucléotides d'une taille comprise entre *l-4* et *l+4*.

      - A partir de la matrice d'oligonucléotides, nous calculons la température de fusion de chaque oligonucléotide présent dans la matrice, selon la formule des proches voisins de Santalucia (https://www.pnas.org/doi/full/10.1073/pnas.95.4.1460). Ensuite, nous calculons la moyenne de la Tm de tous les oligonucléotides présents qui nous servira de température à laquelle nous souhaitons que les oligonucléotides se rapproche le plus. Cette matrice est appelée matrice de Tm

      - La dernière matrice sert à appliquer les critères de selection selon la température de fusion. La matrice de longueur va recenser la longueur des oligonucléotides respectant la condition suivante : leur température doit être comprise à +- 3°C de la température moyenne fixée. Si un oligonucléotide ne respecte pas la condition, il sera noté comme "non valide" ou "NaN" dans la matrice.

    La selection des oligonucléotides dans la matrice de longueur correspond non à l'exploration d'une table, mais est assimilé à l'exploration d'un arbre : chaque colonne de la matrice peut être vu comme un niveau de l'arbre tandis que chaque cellule "valide" est un noeud. La lecture démarre à la première position de la séquence d'entrée et on regarde quels oligonucléotides sont valides à cette position. On prend l'un des oligonucléotides valides et on développe jusqu'à la position suivante qui correspond à la position actuelle + la taille de l'oligonucléotide choisi. L'arbre est développé comme ceci jusqu'à trouver une solution. Si aucun oligonucléotide n'est valide à une position, on remonte l'arbre à la position précédente et on prend l'oligonucléotide suivant dans la position.  



2. Cette méthode peut permettre d'obtenur de meilleurs résultats que la méthode naïve car l'ajout de condition est beaucoup plus stricte et facile que dans la méthode naïve. De plus, elle explore à partir de tous les oligonucléotides pouvant être synthétisés, contrairement à la méthode naïve qui ne regarde que les oligonucléotides avec une température extremum.  



3. Selon les modifications que nous avons appliqué à notre algorithme, la complexité en temps de la deuxième méthode est de ***Θ(2^d)*** avec *d* = profondeur moyenne de l'arbre et la complexité en espace sera de ***Θ(2d)*** car à chaque itération, on ne sauvegarde que les noeuds dans la profondeur à laquelle on recherche les solutions.  


    Pour la première méthode, la complexité en temps sera de ***Θ(2ab)*** avec *a* = nombre d'itération de modification des chevauchements d'oligonucléotides et *b* = nombre de blocs. La complexité en espace est de ***Θ((o+n\*a)\*b)*** avec *o* = nombre d'oligonucleotides finaux dans un bloc, *n* = nombre de chevauchements d'oligonucléotides dans un bloc, *a* = nombre d'itération de modification des chevauchements d'oligonucléotides et *b* = nombre de blocs, car on sauvegarde l'état de tous les chevauchements à chaque itération, on sélectionne la meilleure itération et on produit les oligonucleotides finaux à partir des chevauchements.  


    La méthode naïve demande actuellement moins de temps que la deuxième méthode, mais prends plus d'espace à cause des états intermédiaires qu'il faut systématiquement sauvegarder. Nous continuons de chercher de meilleurs algorithmes autre que la seconde méthode pour palier au problème de temps, tout en gardant la possibilité de trouver de meilleurs solutions que la méthode naïve.
