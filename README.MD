# Comparative spatial data analysis on accessibility realities of public transport and private car in Helsinki and Tallinn



Download OSMConvert from [here](https://wiki.openstreetmap.org/wiki/Osmconvert) and install it.

Download the version of OpenTripPlanner suitable for your analysis, find it [here](https://repo1.maven.org/maven2/org/opentripplanner/otp/), "download the file whose name ends with `.shaded.jar`", [see documentation](http://opentripplanner.readthedocs.io/en/latest/Getting-OTP/#getting-opentripplanner).

##Tallinn

Download the GTFS data for Tallinn: http://transitfeeds.com/p/maanteeamet/510/latest, copy the file to the folder ```..\graph-location```

```
    $ ..\osmconvert64-0.8.8p ...\estonia-latest.osm.pbf -b=24.223417,59.244663,25.268491,59.641916 --complete-ways -o=tallinn.pbf
    $ cp tallinn.pbf ..\graph-location\tallinn
```

Build the graph:
```
    $ java -Xmx10G -jar otp-1.3.0-SNAPSHOT-shaded.jar --build ..\graph-location\tallinn --basePath ..\graph-location\tallinn --analyst
```

Start OpenTripPlanner server:
```
    $ java -Xmx10G -jar otp-1.3.0-SNAPSHOT-shaded.jar --router tallinn --graphs ..\graph-location --server 
```

