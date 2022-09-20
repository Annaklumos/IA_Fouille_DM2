# Première partie

1. Un parseur est un analyseur synthaxique. Il permet d'analyser un langage, d'extraire des éléments et les interprete. Ici, notre but est d'utiliser l'outil via un terminal en tapant la commande exécutant l'outil à travers un parseur de la ligne de commande. Nous trouvons que ceci est beaucoup plus simple à utiliser qu'un dictionnaire, car il permet d'avoir un contrôle sur les variables utilisées par le script sans avoir à lire le code pour trouver dans quel(s) module(s) les variables sont créées et appelées.

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
    3. La fonction **design_oligo()** récupère les blocs créés à partir du module ***split*** et de la fonction **create_design_blocks()** et lance **compute_tm()** pour chaque bloc
    4. **compute_tm()** calcule la Tm, classe et modifie chaque set de chevauchements d'oligonucléotides un nombre X de fois et renvoie ensuite la fonction **output_final_oligo()** si les chevauchements respectent les conditions de l'utilisateur, sinon renvoie "False"
    5. **output_final_oligo()** récupère les données de l'itération *i* avec le meilleur delta Tm, crée les oligonucléotides à partir des chevauchements et écrit les oligonucléotides dans des fichiers .tsv/.fasta avec les annotations en .bed

Vous trouverez en pièce jointe un diagramme interactif d'appel des fonctions citées.

Cette implémentation n'est pas la plus optimale, pour avoir un code plus lisible et fonctionnel, il faudrait transformer les oligonucléotides, les chevauchements et les blocs sous forme de classes (par exemple une classe Oligo(), une classe Block() et une classe Overlaps()). Néanmoins, le fait d'utiliser des dictionnaires nous facilite la tâche pour créer une API autour de l'outil, car il est facile de transformer un dictionnaire en une requête/réponse sous format JSON.

3. Actuellement, les tests implémentés permettent de vérifier quelques critères:
  - On souhaite vérifier si les différents fichiers sont bien créés à l'aide de nos fonctions, plusieurs tests sont utilisés pour vérifier ça
  - On souhaite savoir si les blocs ou les oligonucléotides formés à l'aide de l'outil reproduisent bien exactement le gène de référence. Plusieurs tests sont utilisés pour mapper les oligonucléotides/blocs au gène de référence et regarde le taux de couverture. Si le gène n'est pas entièrement couvert, le test est faux.
  - On souhaite tester les différents paramètres automatiquement, plusieurs tests ont étés en utilisant une grille de paramètres pour voir si l'outil ne se termine pas prématurément.


# Deuxième partie

1. La méthode n'est pas vraiment une méthode dynamique, je pense avoir confondu avec une autre méthode assez similaire, mais qui d'après eux repose sur l'algorithme de Viterbi qui utilise de la programmation dynamique (https://www.frontiersin.org/articles/10.3389/fgene.2022.836108/full). La méthode actuelle consiste en l'exploration d'une table sous forme d'un arbre binaire, à la recherche d'une solution selon la fonction d'exploration de l'arbre. 
