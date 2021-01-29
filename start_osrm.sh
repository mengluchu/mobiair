#!/usr/bin/env bash
set -e


osrm/bin/osrm-routed --verbosity=WARNING --algorithm=MLD --threads=1 --port=55001 data/nl_car.osm.pbf &
osrm/bin/osrm-routed --verbosity=WARNING --algorithm=MLD --threads=1 --port=55002 data/nl_bicycle.osm.pbf &
osrm/bin/osrm-routed --verbosity=WARNING --algorithm=MLD --threads=1 --port=55003 data/nl_foot.osm.pbf &
osrm/bin/osrm-routed --verbosity=WARNING --algorithm=MLD --threads=1 --port=55004 data/nl_train.osm.pbf &

osrm/bin/osrm-routed --verbosity=WARNING --algorithm=MLD --threads=1 --port=55010 data/ch_car.osm.pbf &
osrm/bin/osrm-routed --verbosity=WARNING --algorithm=MLD --threads=1 --port=55011 data/ch_bicycle.osm.pbf &
osrm/bin/osrm-routed --verbosity=WARNING --algorithm=MLD --threads=1 --port=55012 data/ch_foot.osm.pbf &
osrm/bin/osrm-routed --verbosity=WARNING --algorithm=MLD --threads=1 --port=55013 data/ch_train.osm.pbf &


