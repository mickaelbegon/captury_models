# Feuille de route scientifique des comparaisons cinematiques

## Objectif

Rendre comparables, dans un meme cadre biomecanique documente, quatre
representations de la cinematique:

1. le squelette Captury markerless exporte en BVH ou FBX;
2. les angles articulaires exportes dans le C3D Captury;
3. le squelette Motive Full Body 57 exporte en BVH ou FBX;
4. le modele BioBuddy/biorbd construit et reconstruit par cinematique inverse
   a partir des marqueurs Motive.

Motive BVH/FBX et BioBuddy ne sont pas deux mesures experimentales
independantes: ils utilisent les memes trajectoires de marqueurs. Leur
comparaison isole principalement les effets de la definition du modele, des
reperes segmentaires et de la cinematique inverse.

La sortie cible n'est pas seulement un ensemble de courbes. Chaque resultat
doit etre accompagne de la convention utilisee, de son statut ISB, de la
qualite du recalage et de l'incertitude de reconstruction.

## Fonctionnalites deja disponibles

- inventaire et appariement des C3D, BVH et FBX Captury/Motive;
- extraction des `q` BVH/FBX au moyen de `parser.to_q()`;
- separation des marqueurs et des canaux angulaires Captury dans le C3D;
- gestion des unites, de l'axe vertical et des translations de racine;
- construction du modele BioBuddy Motive 57 avec statique, SCoRE et SARA;
- IK non lineaire TRF BioBuddy et lancement batch de la cinematique inverse;
- calcul des centres articulaires, dimensions et rotations segmentaires;
- comparaison de series temporelles et metriques MAE, RMSE, biais, CCC et
  correlation;
- visualisation 3D des marqueurs, chaines et reperes locaux;
- cache par essai et structure preparant les analyses multi-participants.

## Limites scientifiques confirmees

### Comparaison directe des q

Un nom identique tel que `LeftUpLeg_rotX` ne garantit pas une grandeur
biomecanique identique. Les axes locaux, l'origine, le sens des axes, la
sequence Euler et la convention active/passive peuvent differer entre Captury,
Motive et BioBuddy.

La representation canonique doit donc etre une transformation rigide ou une
matrice de rotation. Le contrat cible utilise des vecteurs colonnes et note
`R_lab_segment` la matrice dont les colonnes sont les axes du segment exprimes
dans le laboratoire. Avec ce contrat uniquement, les angles articulaires sont
extraits apres harmonisation des reperes proximal et distal a partir de:

```text
R_proximal_distal = R_lab_proximal.T @ R_lab_distal
```

### Reexpression ZXY

La reexpression actuelle extrait des angles Euler depuis les rotations
globales des segments. Elle est utile comme diagnostic de repere segmentaire,
mais elle ne constitue pas encore une cinematique articulaire. Elle extrait
actuellement les angles directement a partir de:

```text
R_lab_segment
```

La cinematique cible doit utiliser la rotation relative definie ci-dessus,
apres application des corrections de reperes propres aux deux segments.

### Recalage circulaire

Les centres articulaires ne doivent pas servir simultanement a estimer le
recalage et a mesurer l'erreur de ces memes centres. Une transformation
statique devra etre estimee avec des reperes independants ou avec un jeu de
centres reserve a la calibration, puis figee pour tous les essais dynamiques.

### Angles C3D Captury

Le C3D P6 annonce treize triplets angulaires en degres. Plusieurs composantes
epaule/coude/poignet sont toutefois exactement dupliquees dans plusieurs
essais. Ces canaux doivent rester une source Captury distincte tant que les
informations suivantes ne sont pas confirmees:

- articulation et segments proximal/distal;
- ordre et nature de la sequence Euler/Cardan;
- correspondance des trois composantes avec les plans anatomiques;
- signes, zero anatomique et convention droite/gauche;
- rotations relatives ou absolues;
- validite des composantes dupliquees ou constantes.

### Statistiques

Les frames ne sont pas des observations independantes. Les analyses de
population devront produire une valeur par essai, puis par participant. Les
correlations et NRMSE ne seront pas interpretees sur un essai statique ou une
DoF dont l'amplitude est proche de zero.

## Grille de deviations ISB

La grille reprend les categories D1-D6 proposees dans l'article Spartacus:

- **D1**: orientation ou direction des axes du repere local;
- **D2**: methode de construction des axes et landmarks utilises;
- **D3**: origine du repere local;
- **D4**: sequence utilisee pour exprimer les rotations articulaires;
- **D5**: repere utilise pour exprimer les translations articulaires;
- **D6**: calcul thoracohumeral utilise pour decrire le mouvement de l'humerus.

Etats utilises dans les rapports futurs:

- `conforme`: conformite demontree par metadata et test numerique;
- `deviation`: difference explicite avec la recommandation retenue;
- `documente_non_evalue`: construction connue mais non confrontee au standard;
- `partiel`: fonctionnalite logicielle disponible, sans revendication de
  conformite ISB partielle;
- `inconnu`: information absente ou proprietaire;
- `non_applicable`: critere sans objet pour cette articulation.

### Etat initial

| Source ou region | D1 | D2 | D3 | D4 | D5 | D6 |
|---|---|---|---|---|---|---|
| Captury BVH/FBX | inconnu | inconnu | inconnu | inconnu | non_applicable | inconnu pour l'epaule; non_applicable ailleurs |
| Captury angles C3D | inconnu | inconnu | inconnu | inconnu | non_applicable | inconnu pour l'epaule; non_applicable ailleurs |
| Motive BVH/FBX | inconnu | inconnu | inconnu | inconnu | non_applicable | inconnu pour l'epaule; non_applicable ailleurs |
| BioBuddy pelvis/hanche | documente_non_evalue | documente_non_evalue | documente_non_evalue | documente_non_evalue | non_applicable | non_applicable |
| BioBuddy cuisse/genou | documente_non_evalue | documente_non_evalue (SARA) | documente_non_evalue | documente_non_evalue | non_applicable | non_applicable |
| BioBuddy jambe/cheville/pied | documente_non_evalue | documente_non_evalue | documente_non_evalue | documente_non_evalue | non_applicable | non_applicable |
| BioBuddy thorax/rachis | documente_non_evalue | documente_non_evalue | documente_non_evalue | documente_non_evalue | non_applicable | non_applicable |
| BioBuddy epaule | documente_non_evalue | documente_non_evalue | documente_non_evalue | documente_non_evalue | non_applicable | deviation D6 documentee: Thorax vers UpperArm sans scapula |
| BioBuddy coude/poignet | documente_non_evalue | documente_non_evalue | documente_non_evalue | documente_non_evalue | non_applicable | non_applicable |

Cette table est un etat d'audit, pas une certification. Une orientation `Y-up`
ou une sequence `ZXY` ne suffit pas a conclure a la conformite ISB. Le genou
doit par ailleurs etre evalue selon un standard explicitement choisi, par
exemple le JCS de Grood-Suntay, car la partie I des recommandations ISB traite
principalement la cheville, la hanche et le rachis.

### Completude des modeles, separee de D1-D6

Les nombres de segments et de DoF sont des limites du modele, pas des
deviations D4. Le template BioBuddy courant doit donc rapporter separement:

- genou: rotation de jambe reduite a `Z`;
- cheville/pied: rotations du pied reduites a `ZX`;
- coude/avant-bras: rotations reduites a `ZY`;
- poignet/main: rotations reduites a `ZX`;
- epaule: absence de scapula et articulation directe `Thorax -> UpperArm`.

L'absence de scapula rend impossible le calcul glenohumeral. Le calcul direct
Thorax vers UpperArm est classe D6 lorsqu'il est interprete comme angle
d'epaule, tout en restant une pratique utile pour decrire le mouvement
thoracohumeral.

## Protocole de validation par phase

Chaque phase suit obligatoirement cet ordre:

1. ecrire des tests de caracterisation avant le changement;
2. enregistrer les resultats de reference et les tolerances numeriques;
3. effectuer une seule modification scientifique clairement bornee;
4. lancer les tests unitaires, tests d'integration et controles numeriques;
5. faire revoir le diff et les preuves par un agent validateur independant;
6. corriger les objections ou documenter explicitement les limites;
7. mettre a jour ce document et le README avant le commit de la phase.

Un gate est refuse si les unites, axes, sens de multiplication, sequence,
convention droite/gauche ou donnees utilisees pour la calibration ne sont pas
identifiables dans le rapport genere.

### Checklist de gates independants

| Gate | Objet | Etat initial | Preuve requise pour accepter |
|---|---|---|---|
| G0 | Provenance | partiel | manifeste des fichiers, logiciels, versions, frequences, unites et source BVH/FBX |
| G1 | Classification C3D | partiel | table label/type/unite/definition et tests metadata, doublons, NaN |
| G2 | Angles Captury | refuse | parent/distal, sequence, convention, signes et pose neutre documentes |
| G3 | Unites et axes | partiel | longueurs plausibles, matrices orthonormales et determinant positif |
| G4 | Translation racine | partiel | rapport Captury/Motive separe avec offset lu et politiques forcees |
| G5 | Recalage spatial | accepte avec centres reserves; validation anatomique en attente | calibration statique figee; centres de calibration exclus des metriques principales |
| G6 | ISB D1-D3 | accepte pour diagnostic BioBuddy | cibles Wu versionnees, roundtrip bioMod et evaluation statique; Captury/Motive proprietaires restent inconnus |
| G7 | ISB D4 rotations | accepte pour diagnostic BioBuddy hanche/genou; final ISB refuse | matrices parent-enfant corrigees, axes JCS, signes, singularites, roundtrip et provenance; G6 et corrections des autres sources/articulations restent incomplets |
| G8 | ISB D5 translations | non applicable actuellement | translation articulaire distale exprimee dans le repere proximal ISB si elle est analysee |
| G9 | BioBuddy dynamique | partiel: statique reconstruit, dynamique refuse | IK de chaque essai, residus et absence de fallback silencieux |
| G10 | Synchronisation | accepte pour lag constant diagnostique; derive et contacts force-plate non valides | lag, evenements communs, erreur residuelle et cycles documentes |
| G11 | Metriques | partiel | SO(3), waveform, ROM, timing et agregation essai puis participant |
| G12 | Rapport reproductible | a construire | tableau D1-D6 avec preuve associee a chaque statut |

Un gate `partiel` signifie que du code existe mais que la preuve scientifique
reste incomplete. Un gate `refuse` interdit son utilisation dans une
conclusion d'accord biomecanique; les graphes correspondants peuvent rester
disponibles comme diagnostics explicitement marques.

Le statut D5 `non_applicable` signifie qu'aucune translation articulaire n'est
actuellement comparee. La translation de racine ne constitue pas une
translation articulaire au sens D5.

## Phases de travail

### Phase 0 - Contrats et donnees de reference

**Etat:** contrat implemente; gate G0 encore partiel.

Creer un schema versionne `kinematic_conventions.json` contenant, pour chaque
source, segment et articulation:

- parent, enfant et aliases de noms;
- definition de l'origine et landmarks;
- axes locaux, sens positif et handedness;
- repere laboratoire et unite;
- convention des matrices et transformations;
- sequence Euler, intrinseque/extrinseque, active/passive;
- signification anatomique, zero et signe de chaque DoF;
- provenance de l'information et niveau de confiance;
- fichiers sources, version du logiciel exporteur et version des dependances de
  lecture/reconstruction.

**Tests avant modification:** validation JSON, champs obligatoires, aliases
uniques, graphe parent-enfant sans cycle et matrices orthonormales.

**Gate:** aucune comparaison angulaire finale ne peut utiliser une source dont
la convention requise reste `inconnu`.

**Implemente le 2026-08-12:** `kinematic_conventions.json` inventorie les quatre
representations, leurs segments, articulations, statuts D1-D6 et conventions
connues ou inconnues. `kinematic_conventions.py` valide le schema, les aliases,
le graphe parent-enfant et les rotations de correction, puis expose les raisons
qui bloquent une comparaison finale. Le batch ecrit `provenance_manifest.json`
avec les fichiers effectivement selectionnes, leurs hashes SHA-256, la commande,
le runtime et le statut `diagnostic_only`. G0 reste partiel tant que les
frequences C3D, les unites declarees par fichier et la version des logiciels
exporteurs ne sont pas toutes extraites automatiquement.

### Phase 1 - Representation canonique des rotations

**Etat:** representation SO(3) et audit BVH/FBX implementes; gate refuse sur
les donnees P6 Static.

Extraire les matrices globales de segment pour les quatre sources, dans un
contrat commun `R_lab_segment` et `T_lab_segment`. Comparer BVH et FBX d'un meme
systeme avant de les traiter comme interchangeables.

**Tests avant modification:** identite, rotations elementaires, composition,
inversion, conversion model Y-up vers C3D Z-up, orthonormalite et determinant
positif; roundtrip parser -> matrices -> modele.

**Gate:** l'ecart geodesique BVH/FBX est rapporte par segment et essai. Tout
ecart au-dessus de la tolerance bloque le choix automatique de la source.

**Implemente le 2026-08-12:** `kinematic_rotations.py` impose le contrat
`R_lab_segment`, projette uniquement les faibles residus numeriques FBX sur
SO(3), rejette les reflexions, interpole les rotations par SLERP et calcule la
deviation geodesique ainsi que son vecteur logarithmique. Le registre associe
les noms BVH/FBX a 15 identifiants anatomiques communs. Le CLI
`--audit-bvh-fbx-rotations` ecrit un resume JSON et les series temporelles dans
un NPZ compresse. Le mode `--model-source auto` est bloque lorsque la
couverture est incomplete ou qu'un p95 segmentaire depasse
`--bvh-fbx-max-p95-geodesic-deg`.

Les horodatages des deux exports sont actuellement ramenes au temps ecoule
depuis leur premier echantillon, puis le FBX est interpole par SLERP sur les
temps BVH communs. Aucun lag ni frame initiale manquante n'est estime: un
decalage residuel peut donc provoquer un refus conservateur sur un essai
dynamique et devra etre traite par G10.

Sur P6 Static avec une tolerance diagnostique de 5 degres, Captury est bloque
sur 14/15 segments (p95 maximal 178.8 degres) et Motive sur 15/15 segments
(p95 maximal 6.9 degres). Ces resultats prouvent que les exports ne sont pas
interchangeables dans leur repere natif; ils ne permettent pas encore
d'identifier si l'ecart provient des axes locaux, d'une convention passive,
d'une correction exporter ou de la sequence de reconstruction. Le seuil de
5 degres est un gate d'ingenierie configurable et ne constitue ni une
tolerance ISB ni une validation anatomique.

### Phase 2 - Recalage spatial non circulaire

**Etat:** protocole a centres reserves implemente; G5 accepte pour la
comparaison diagnostique Captury vers Motive des centres non reserves. La
validation anatomique reste en attente car les centres de calibration sont
derives des modeles et non des landmarks mesures independamment.

Separer les transformations suivantes dans leur ordre reel d'application:

1. interpretation de la translation de racine dans les `q` natifs;
2. cinematique directe dans le repere modele natif;
3. conversion unite et axes du fichier vers le C3D;
4. calibration statique source vers laboratoire Motive;
5. comparaison dynamique sans nouveau recalage.

**Tests avant modification:** transformations synthetiques connues, invariance
des distances, ordre de composition et comparaison avec/sans un landmark
reserve.

**Gate:** les centres servant de variable de resultat ne sont pas utilises
pour ajuster la transformation qui minimise leur propre erreur. Les erreurs
avant et apres chaque etape sont conservees.

**Implemente le 2026-08-12:** le mode par defaut ajuste une transformation
rigide sur `Hips`, `Head`, `LeftShoulder` et `RightShoulder` du seul essai
`Static`. Ces quatre centres sont ensuite exclus de
`joint_centre_metrics.csv` et conserves dans le diagnostic distinct
`alignment_calibration_centre_metrics.csv`. Les 18 autres centres communs
constituent le jeu d'evaluation. Les transformations Captury vers Motive et
Motive vers C3D ainsi que les politiques de translation racine Captury/Motive
sont serialisees dans `spatial_calibration.json`, puis reutilisees sans
reestimation sur chaque essai dynamique. Le mode
`legacy_all_centres` demeure disponible comme diagnostic explicitement
circulaire.

Sur le smoke P6 BVH `Static`/`Marche_001`, la mediane statique est de
19.7 mm sur les quatre centres de calibration et de 44.8 mm sur les 18 centres
tenus a l'ecart. Les deux matrices et les deux translations du rapport
`Marche_001` sont numeriquement identiques a celles du statique; les politiques
de translation racine sont egalement figees. Le recalage Motive vers C3D
utilise encore des proxies issus des marqueurs Motive. Comme cette seconde
transformation est appliquee en commun aux centres Captury et Motive, elle ne
modifie pas leur distance paire a paire, mais sa validite anatomique reste a
examiner pour les comparaisons de marqueurs et la visualisation.

### Phase 3 - Reperes segmentaires anatomiques et ISB D1-D3

**Etat:** phase terminee le 2026-08-12; gate G6 acceptee pour un audit
diagnostique BioBuddy, avec indisponibilites explicites et sans revendication
de conformite D1 fondee sur un seuil arbitraire.

Construire une matrice de correction source vers repere anatomique pour chaque
segment et chaque cote. Pour BioBuddy, confronter les definitions du template
Motive 57 aux landmarks ISB. Pour Captury et Motive, classer les definitions
inaccessibles comme `inconnu` plutot que de supposer leur conformite.

**Tests avant modification:** reperes droits, directions craniale/anteriorite/
lateralite, symetrie gauche-droite et stabilite du repere statique.

**Gate:** rapport D1-D3 par segment avec preuve, deviation angulaire et origine
en millimetres.

**Implemente le 2026-08-12:** `isb_segment_audit.py` produit une ligne D1-D3
pour chaque source et segment dans `isb_d1_d3_audit.json` et dans une table
compacte `isb_d1_d3_audit.npz`. Pour BioBuddy, le module introspecte le
template Motive 57 effectivement importe et serialise l'origine, les deux axes
bruts, l'axe conserve, les essais fonctionnels SCoRE/SARA et leurs fallbacks.
Le chemin et le SHA-256 du template importe sont enregistres afin de detecter
les copies locales divergentes. Pour Captury et Motive, les definitions
proprietaires restent `inconnu`.

`isb_segment_frames.json` versionne maintenant les cibles Wu 2002/2005, le DOI,
la section, l'option retenue, l'origine et les axes. La comparaison symbolique
BioBuddy classe les differences D2/D3 demontrees et garde D1
`documente_non_evalue` quand une direction anatomique ne suffit pas a prouver
la conformite. La creation Motive 57 ecrit un roundtrip
`<modele>.roundtrip.json` des RT locaux template -> bioMod et une evaluation
`<modele>.isb_static.json`. Sur P6, le roundtrip couvre les 15 segments; 7
reperes ont une deviation angulaire, une distance d'origine en millimetres,
les determinants et les erreurs d'orthogonalite. Thorax, tete, mains, pieds et
bras superieurs restent explicitement indisponibles: T8 absent (TV7 n'est pas
substitue), aucune cible tete selectionnee, HM2 n'est pas substitue au
troisieme metacarpien, et les postures neutre de cheville et humerale option 2
ne sont pas validees dans les metadonnees P6.
Captury et Motive export restent `inconnu` faute de definition proprietaire.

### Phase 4 - Cinematique articulaire et ISB D4-D6

**Etat:** phase terminee le 2026-08-12; G7 accepte comme diagnostic pour les
hanches et genoux BioBuddy disposant de deux corrections de repere justifiees.
G7 final ISB reste refuse tant que les repères D1-D3 ne sont pas valides comme
conformes et que les autres sources restent proprietaires. Les matrices
relatives restent diagnostiques pour Captury/Motive et les autres articulations.

Calculer les rotations parent-enfant corrigees puis extraire une convention
commune propre a chaque articulation. Les matrices/quaternions restent
disponibles pour une mesure geodesique independante de la sequence Euler.

**Tests avant modification:** flexion pure, abduction pure, rotation axiale,
cas combines, singularites, unwrap, droite/gauche et roundtrip matrice-Euler.

**Gate:** chaque courbe indique explicitement proximal, distal, sequence,
signes et unite. L'epaule distingue thoracohumeral et glenohumeral; D6 est
rapporte separement. Un angle glenohumeral n'est pas produit lorsque le modele
ne contient pas de scapula.

**Implemente le 2026-08-12:** `isb_joint_kinematics.json` versionne les cibles
Wu/Grood-Suntay, les segments proximal/distal, les sequences, les composantes,
les signes et le traitement gauche/droite. `joint_kinematics.py` applique les
corrections par multiplication droite, calcule
`R_proximal_distal = R_lab_proximal.T @ R_lab_distal`, extrait les angles
Euler/Cardan intrinseques, matérialise les axes fixe proximal, flottant et fixe
distal du JCS, applique les signes explicites, effectue l'unwrap et conserve
les flags de singularite ainsi que l'erreur de roundtrip geodesique. La translation D5 est
definie et testee comme difference d'un point commun exprimee dans le repere
proximal, mais reste `non_applicable` dans le batch faute de translation
articulaire reconstruite.

Le batch ecrit `joint_kinematics_d4_d6.json` et un NPZ compresse. Sur le smoke
P6 Static de 20 frames, BioBuddy produit les hanches et genoux droit/gauche en
ZXY, avec les corrections et le SHA-256 du sidecar statique enregistres, sans
singularite et avec une erreur maximale de roundtrip inferieure a
`2.3e-14 deg`. Captury et Motive conservent leurs 13 rotations relatives SO(3)
mais aucune composante Euler anatomique n'est emise car leurs corrections de
repere source vers ISB restent inconnues. L'epaule BioBuddy est explicitement
`thoracohumeral` avec D6 `deviation`; aucun angle glenohumeral n'est invente en
l'absence de scapula.

La revue independante finale accepte ce gate diagnostique. Elle maintient G7
final ISB refuse parce que les corrections anatomiques proviennent de l'audit
G6 diagnostique. Le JSON D4-D6 conserve les matrices proximal/distal appliquees,
le chemin et le SHA-256 du sidecar statique, ainsi que le SHA-256 du bioMod; le
manifeste final hashe egalement les sorties JSON/NPZ.

### Phase 5 - Decodeur des angles C3D Captury

Creer une table de mapping specifique aux treize canaux Captury et verifier les
metadata fournisseur ou un jeu de mouvements controles. Detecter les canaux
dupliques, constants, discontinus ou hors plage.

**Tests avant modification:** C3D synthetique, metadata `POINT:ANGLES`, unite
degre/radian, duplication exacte, composante constante et mouvements
uniplanaires controles.

**Gate:** un canal non decode reste visible comme donnees brutes, mais est
exclu des metriques d'accord anatomique.

**Etat au 2026-08-12:** gate diagnostique atteint. Le registre versionne
`captury_c3d_angles.json` relie les 13 abbreviations `POINT:LABELS`, les noms
longs `POINT:ANGLES` et les articulations. Il ne definit volontairement ni
sequence Euler, ni signes, ni semantique anatomique X/Y/Z. Le decodeur
`captury_c3d_angles.py` ignore `POINT:UNITS=mm` pour les angles, enregistre
`deg`/`rad` comme hypothese CLI lorsqu'aucune metadata dediee n'existe et ecrit
`captury_c3d_angle_decode.json`. Les courbes brutes restent visibles avec
`eligible_for_anatomical_agreement=false`.

Le smoke P6 Static identifie 13 canaux, trois groupes de composantes exactement
dupliquees et deux composantes constantes (`left_elbow/Y`, `left_wrist/Z`).
Les essais synthetiques couvrent metadata, degres/radians, duplication,
constance, discontinuite, plage et mouvement uniplanaire. La conformite
anatomique reste refusee jusqu'a obtention d'une documentation fournisseur ou
d'un protocole controle permettant d'identifier sequence, signes et axes.

### Phase 6 - BioBuddy comme troisieme modele dynamique

Executer l'IK non lineaire TRF statique puis le batch sur tous les essais Motive. Enregistrer
les residus marqueurs, les marqueurs utilises, les echecs, les centres SCoRE,
les axes SARA et leurs incertitudes.

**Tests avant modification:** chargement du modele, mapping Motive 57,
marqueurs manquants, reconstruction synthetique et lecture des `q`/matrices.

**Gate:** aucun fallback silencieux BioBuddy vers Motive. Une source absente
est affichee comme absente et la raison apparait dans le rapport.

**Etat au 2026-08-12:** implementation fonctionnelle et gate technique atteint;
validation scientifique dynamique encore refusee.
Le batch BioBuddy utilise directement les 47 marqueurs techniques du BioMod,
sans l'ancien pipeline IK BVH/FBX. Le diagnostic a montre que l'erreur historique
d'environ 58,7 mm provenait principalement d'un melange entre indices de tous
les marqueurs et indices des marqueurs techniques apres insertion de marqueurs
anatomiques. Le smoke P6 Static corrige donne 0,350 mm d'erreur moyenne sur 10
frames. Les resultats sont caches par contenu BioMod/C3D, solveur et parametres;
le Static post-creation est reutilise. Temps, `nfev`, residus et ETA sont traces.
BioBuddy alimente dimensions, centres et rotations segmentaires. La comparaison
directe des `q` reste refusee sans mapping anatomique explicite des noms de DoF.

Sur le Static complet, les 773 frames sont reconstruites en environ 1,03 s avec
0,192 mm de residu moyen in-sample sur les marqueurs optimises, 2 572
evaluations de fonction et 47/47 marqueurs techniques. Cette valeur est une
qualite d'ajustement, pas une erreur de validation independante. `Marche_001`
converge numeriquement sur 957/957 frames en environ
8,83 s et 21 138 evaluations, mais conserve 34,95 mm d'erreur moyenne
(RMSE 44,89 mm; p95 94,71 mm). La convergence TRF ne valide donc pas la
cinematique dynamique.

Le controle mecanique a ajuste independamment les clusters rigides du Static au
premier frame de marche. Pour les segments ayant au moins trois marqueurs, la
RMSE est comprise entre 0,23 et 6,15 mm; une erreur globale d'unite ou une
deformation massive des clusters n'explique donc pas le residu de chaine. Les
residus BioBuddy moyens les plus eleves concernent `LUpperArm` (91,93 mm),
`RHand` (87,32 mm), `LHand` (85,09 mm) et `RUpperArm` (84,92 mm), contre
10,08-12,51 mm pour les pieds. Les prochains travaux scientifiques doivent
examiner les centres SCoRE/SARA, les longueurs et liaisons intersegmentaires et
la pauvrete des clusters de bras (un seul marqueur technique par bras) avant
toute acceleration ou interpretation des angles dynamiques.

La revue independante accepte le gate technique apres ajout du hash du code IK
dans la cle, du SHA-256 et du controle de schema du NPZ avant reutilisation,
ainsi que de tests explicites des conversions C3D mm -> BioMod m -> residus mm.
G9 reste partiel: le statique est reconstruit, mais la cinematique dynamique
reste refusee tant que les residus de chaine ne sont pas expliques.

### Phase 7 - Synchronisation et selection des phases

Estimer le decalage temporel entre systemes avant interpolation. Utiliser les
contacts, pics ou signaux communs, puis normaliser les cycles ou phases a
0-100 % lorsque pertinent.

**Tests avant modification:** lag synthetique, frequences differentes,
echantillons manquants, fenetres manuelles et detection des contacts.

**Gate:** lag, methode, erreur residuelle et fenetre analysee sont sauvegardes.

**Etat au 2026-08-13:** phase terminee; gate technique accepte pour un lag
constant diagnostique et une phase selectionnee. La derive d'horloge et la
validation des contacts par plateforme de force restent hors de ce gate.
Motive est l'horloge de reference et la convention est
`captury_corrected_time = captury_original_time + lag_s`. L'estimateur utilise
la correlation d'une vitesse composite sans translation construite sur les
centres articulaires communs apres recalage spatial. Il estime uniquement un
offset constant et refuse le Static, un signal insuffisant, une correlation
inferieure a 0,5, un gain de correlation inferieur a 0,01 ou un optimum a la
limite de recherche. La proeminence du meilleur pic est egalement comparee au
second pic separe afin de refuser un mouvement periodique ambigu; la resolution
de recherche constitue l'incertitude temporelle minimale rapportee. Le meme temps Captury
corrige est utilise pour centres, q, rotations segmentaires, angles C3D et
marqueurs cutanes. Aucune valeur n'est extrapolee hors du recouvrement temporel;
les echantillons correspondants restent manquants dans les sorties et le viewer.
Le JSON sauvegarde methode, lag applique et estime, correlations, proeminence,
resolution, RMSE normalisee et fenetres. Les NPZ sauvegardent le signal
inspectable, la phase composite et les series scientifiques normalisees a
0-100 % pour chaque famille disponible parmi centres, q, rotations
segmentaires et marqueurs apparies. Une famille indisponible reste absente.
Chaque signal publie couverture finie, plus grand gap, gap maximal accepte et
statut; une longue occlusion interne rend la normalisation indisponible au lieu
d'etre traversee par interpolation.
`contact_cycles.json` recense les cycles derives des contacts cinematiques et
les qualifie explicitement de diagnostiques sans validation force-plate. Un
groupe de marqueurs de pied absent rend le cote indisponible et les cycles de
moins de 0,2 s sont rejetes.

Sur P6 `Marche_001`, le meilleur lag est `-8,353 ms`, mais le gain de
correlation n'est que `1,13e-5` (`0,997284` vers `0,997295`): l'algorithme le
refuse comme gain negligeable et applique `0 s`. Le smoke headless produit 101
points pour la phase composite et 16 059 lignes normalisees scientifiques.
Apres rejet des micro-cycles inferieurs a 0,2 s, 8 cycles diagnostiques restent.
Ces chiffres valident la plomberie et le refus
conservateur; ils ne valident pas anatomiquement les contacts ou les cycles.

### Phase 8 - Metriques et incertitudes

Produire par articulation et composante:

- courbes source et difference;
- erreur geodesique de rotation;
- biais, MAE, RMSE, limites d'accord et ROM;
- decalage des extrema et similarite de forme;
- sensibilite aux conventions, recalages et definitions de centre;
- residus IK et qualite des calibrations fonctionnelles.

**Tests avant modification:** cas identiques, biais constant, gain, lag,
inversion de signe, faible amplitude et donnees manquantes.

**Gate:** aucune correlation/NRMSE n'est affichee sans amplitude minimale;
aucune statistique de population ne traite les frames comme sujets.

### Phase 9 - Rapport ISB et interface

Ajouter dans le GUI et dans les rapports:

- tableau D1-D6 filtrable par source, segment et articulation;
- provenance et niveau de confiance de chaque convention;
- avertissements bloquants pour les comparaisons non harmonisees;
- comparaison cote a cote Captury BVH/FBX, Captury C3D, Motive et BioBuddy;
- export compact des matrices, metadata et series dans des fichiers `.npz` et
  JSON versionnes.

**Tests avant modification:** rendu avec sources manquantes, statuts ISB,
fallback interdit et roundtrip des rapports.

**Gate:** le rapport permet a un lecteur externe de reproduire chaque courbe
sans consulter le code de la GUI.

## Jeux de validation minimum

- `Static`: axes, zero, dimensions, origine et recalage seulement;
- `LHip` et `RHip`: centre SCoRE et rotations de hanche;
- `LKnee` et `RKnee`: axe SARA et flexion dominante;
- `LAnkle` et `RAnkle`: centre fonctionnel et cheville;
- `Marche_001`: synchronisation, contacts et cycles;
- `Squat_001`: grandes amplitudes sagittales;
- `THT_001`: epaule et question D6;
- simulations synthetiques: rotations pures et transformations connues.

## References scientifiques

- Wu et al. (2002), recommandations ISB partie I: cheville, hanche et rachis,
  DOI `10.1016/S0021-9290(01)00222-6`.
- Wu et al. (2005), recommandations ISB partie II: epaule, coude, poignet et
  main, DOI `10.1016/j.jbiomech.2004.05.042`.
- Grood et Suntay (1983), JCS applique au genou, DOI
  `10.1115/1.3138397`.
- Spartacus (2025), identification des deviations D1-D6, DOI
  `10.1016/j.jbiomech.2025.112642`.

## Journal des validations

| Date | Phase | Validateur | Resultat | Preuves ou objections |
|---|---|---|---|---|
| 2026-08-12 | Audit initial | Codex | plan redige | Inspection du code, du template Motive 57 et des C3D P6 |
| 2026-08-12 | Audit initial | agent independant Dirac | refuse pour interpretation finale | G2, G5, G7 et G9 refuses; 32 tests cibles reussis dans `bvh-c3d-biobuddy`; `pytest` absent de `captury_biobuddy` |
| 2026-08-12 | Revue du plan | agent independant Dirac | corrections demandees | D4 separe de la completude, statuts ISB non suraffirmes et contrat matriciel explicite |
| 2026-08-12 | Phase 0, tests avant changement | Codex | 12 tests reussis | validation du schema, heritage Motive independant, aliases, cycles, rotations, blockers D1-D4 et manifeste dans `tests/test_kinematic_conventions.py` |
| 2026-08-12 | Phase 0, integration batch | Codex | 171 tests du depot reussis | `provenance_manifest.json`, selection explicite BVH/FBX, essai statique implicite et statut `diagnostic_only`; G0 conserve comme partiel |
| 2026-08-12 | Phase 0, premiere revue | agent independant Sagan | refuse | BVH IK secondaire et artefacts derives utilises dans le batch absents du manifeste |
| 2026-08-12 | Phase 0, revue apres corrections | agent independant Sagan | approuve | Smoke P6 `Static`: quatre entrees et trois artefacts derives hashes; aucun finding bloquant |
| 2026-08-12 | Phase 1, premiere revue | agent independant Laplace | refuse | cache sans registre scientifique, hypothese temporelle implicite, contrat `globalJCS` et roundtrip NPZ non testes, artefacts d'audit absents de la provenance |
| 2026-08-12 | Phase 1, revue apres corrections | agent independant Laplace | approuve | 199 tests du depot, mini-`bioMod` connu, cache version 5, NPZ relu et smoke P6 `Static`; aucun finding bloquant |
| 2026-08-12 | Phase 2, tests et smoke | Codex | reussi | 211 tests du depot; P6 `Static` puis `Marche_001`; matrices et politiques statiques figees, centres reserves absents des metriques principales, cache relu et calibration hashee dans la provenance |
| 2026-08-12 | Phase 2, premiere revue | agents independants Hooke et Singer | refuse | fingerprint incomplet, ordre des transformations mal documente, cache avec CSV vide, provenance et option GUI a corriger |
| 2026-08-12 | Phase 2, revue apres corrections | agents independants Hooke et Singer | approuve | aucun finding bloquant; limites anatomiques explicites; Black, py_compile et validations ciblees reussis |
| 2026-08-12 | Phase 3, cadrage D1-D3 | agent independant Copernicus | approuve sous conditions | audit traçable accepte; exige distinction definition/fonctionnel/fallback/repere bioMod et interdit toute conformite sans cible ISB versionnee et preuve numerique |
| 2026-08-12 | Phase 3, fermeture G6 | agents independants Maxwell et Meitner | approuve sous conditions explicites | cibles Wu verifiees; roundtrip template-bioMod valide; aucune substitution T8/TV7 ou HM2/MC3 et Captury/Motive proprietaires maintenus inconnus |
| 2026-08-12 | Phase 3, premiere revue d'implementation | agent independant Pascal | refuse | preuve D1-D3 par segment absente, indisponibilite du template non propagee aux lignes et exception occlusions-only mal documentee |
| 2026-08-12 | Phase 3, revue apres corrections | agent independant Pascal | approuve | preuve par segment liee au SHA-256, statut runtime explicite, aucune deviation numerique inventee; 221 tests du depot, Black, py_compile, smoke P6 Static et smoke occlusions-only reussis |
| 2026-08-12 | Phase 3, revue scientifique finale | agent independant Euler | approuve pour audit diagnostique BioBuddy | roundtrip strict des 15 segments incluant la hierarchie, sidecars lies au SHA-256, 7/15 evaluations anatomiques et 8 indisponibilites justifiees sans suraffirmation de conformite |
| 2026-08-12 | Phase 4, premiere revue | agent independant Planck | refuse G7 final; accepte partiellement le diagnostic | corrections non tracees, equivalence JCS non testee, signes non appliques et smoke anterieur au diff |
| 2026-08-12 | Phase 4, revue apres corrections | agent independant Planck | accepte G7 diagnostique; refuse G7 final ISB | matrices et SHA traces, axes JCS/signatures testees, signes appliques, singularites/quaternions conserves, smoke final P6; G6 reste diagnostique |
| 2026-08-12 | Phase 5, tests, smoke et revue finale | Codex + agent independant Planck | gate diagnostique approuve | 259 tests, 2 subtests, `black --check`, `py_compile` et smoke P6 Static dans `/tmp/captury_phase5_smoke_final`; 13 identites decodees, unite `deg` explicitement assumee, trois groupes dupliques et deux composantes constantes; JSON/CSV/NPZ traces dans la provenance; accord anatomique refuse |

Chaque prochaine entree de validation doit enregistrer la commande de test,
l'environnement, le commit ou diff examine et le chemin de la sortie brute.
