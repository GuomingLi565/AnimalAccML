This data set contains tri-axial accelerometer sensor data with thirteen different labeled cow behaviors. This data was gathered with a 16bit +/- 2g Kionix KX122-1037 accelerometer attached to the neck of six different Japanese Black Beef Cows at a cow farm of Shinshu University in Nagano, Japan.
	
The data gathering took place over the course of two days in which the cows were allowed to roam freely in two different areas, namely, a grass field and farm pens, while being filmed with Sony FDR-X3000 4K video cameras.

The timestamps of the video and accelerometer data were matched while human observers which included behavior experts and non-experts labeled the data from the video footage.

197 minutes of data comprising thirteen different behaviors were labeled which are the following categorically sorted in amount of samples:
|     | Cow 1 | Cow 2 | Cow 3 | Cow 4 | Cow 5 | Cow 6 | Sum    | Description                     |
|-----|-------|-------|-------|-------|-------|-------|--------|---------------------------------|
| RES | 35814 | 47059 | 20501 | 15735 | 11025 | 19996 | 150130 | Resting in standing position    |
| RUS | 1620  | 25930 | 11156 | 14523 | 0     | 0     | 53229  | Ruminating in standing position |
| MOV | 6376  | 8437  | 7532  | 17248 | 4846  | 5760  | 50199  | Moving                          |
| GRZ | 2416  | 2199  | 0     | 2707  | 2442  | 7849  | 17613  | Grazing                         |
| SLT | 204   | 0     | 10654 | 0     | 0     | 0     | 10858  | Salt licking                    |
| FES | 6809  | 0     | 0     | 0     | 1125  | 0     | 7934   | Feeding in stancheon            |
| DRN | 1176  | 0     | 1300  | 0     | 0     | 0     | 2476   | Drinking                        |
| LCK | 0     | 0     | 649   | 297   | 0     | 356   | 1302   | Licking                         |
| REL | 0     | 360   | 0     | 404   | 0     | 0     | 764    | Resting in lying position       |
| URI | 239   | 0     | 383   | 0     | 0     | 0     | 621    | Urinating                       |
| ATT | 57    | 50    | 0     | 62    | 0     | 197   | 366    | Attacking                       |
| ESC | 0     | 0     | 0     | 128   | 0     | 0     | 128    | Escaping                        |
| BMN | 0     | 54    | 0     | 0     | 0     | 0     | 54     | Being mounted                   |
| Sum | 54710 | 84089 | 52175 | 51104 | 19438 | 34158 | 295674 |                                 |

Accelerometer sampling rate was set to 25Hz.
The data is split into six .csv files which represents each of the 6 cows above. The columns of these files are defined as follows:
| AccX [g]            | AccY [g]            | AccZ [g]            | label [-]        |
|---------------------|---------------------|---------------------|------------------|
| X-axis acceleration | Y-axis acceleration | Z-axis acceleration | labeled behavior |


The  gathering  of  this  data  with  these  cows  was reviewed  and  approved  by  the  Institutional  Animal  Careand Use Committee of Shinshu University.

