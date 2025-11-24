grep -n "^StartTime," /DataLab/botNCC2/NCC2AllSensors.csv | head

SELECT count(nasc.*) FROM NCC2AllSensors_clean.main.NCC2AllSensors_clean AS nasc
WHERE Label LIKE '%Bot%' AND TRIM(Dir) = '->' AND SensorId = 2