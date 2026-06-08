# Spotify Music Analysis — Key Findings

## Dataset
- 113,999 canciones · 114 géneros · 21 variables de audio

---

## 1. Géneros más populares
- **pop-film** (59.28) y **k-pop** (56.95) lideran popularidad promedio
- Géneros de nicho como **chill** y **sad** superan a géneros mainstream como **pop** (47.58)
- Géneros con audiencias globales consolidadas dominan el ranking

---

## 2. Canciones más populares
- **"Unholy"** de Sam Smith & Kim Petras alcanzó popularidad máxima (100)
- **"La Bachata"** de Manuel Turizo aparece en múltiples géneros con popularidad 98
- Las canciones top aparecen en varios géneros simultáneamente, lo que infla su alcance

---

## 3. Perfil de clusters
| Cluster | Energy | Danceability | Valence | Tamaño |
|---|---|---|---|---|
| Hard Rock / Metal | 0.815 | 0.469 | 0.371 | 28,462 |
| Electronic Instrumental | 0.746 | 0.582 | 0.338 | 12,057 |
| Pop / Dance | 0.732 | 0.696 | 0.700 | 37,514 |
| Hip-Hop / Rap | 0.656 | 0.652 | 0.510 | 4,525 |
| Acoustic / Folk | 0.383 | 0.527 | 0.395 | 23,925 |
| Classical / Ambient | 0.176 | 0.346 | 0.184 | 7,516 |

- **Pop / Dance** es el cluster más grande (37,514 canciones) con el valence más alto (0.700)
- **Classical / Ambient** tiene la energy más baja (0.176) — canciones más tranquilas del dataset

---

## 4. Canciones trending (popularity > 80)
- **Pop** domina con 114 canciones trending
- **Reggae** tiene el promedio más alto entre los trending (87.0)
- **Reggaeton** y **latino** aparecen con fuerza, reflejando el auge de la música latina global

---

## 5. Canciones explícitas vs no explícitas
| Tipo | Popularidad avg | Danceability | Energy |
|---|---|---|---|
| Explícita | 36.45 | 0.636 | 0.721 |
| No explícita | 32.94 | 0.560 | 0.634 |

- Las canciones **explícitas son más populares** en promedio (+3.5 puntos)
- También tienen **mayor danceability (+13%)** y **mayor energy (+13%)**
- Solo representan el 8.5% del dataset (9,747 de 113,999)

---

## 6. Metodología
- **Clustering:** KMeans con k=6, features normalizadas con StandardScaler
- **Selección de k:** Elbow Method + Silhouette Score (mejor en k=6, score: 0.210)
- **Visualización:** PCA 2D para reducción de dimensiones
- **Base de datos:** PostgreSQL con SQLAlchemy para consultas estructuradas