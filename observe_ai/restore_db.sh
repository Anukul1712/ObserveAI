#!/bin/bash
echo "Restoring Neo4j database from dump..."
docker-compose run --rm neo4j neo4j-admin database load neo4j --from-path=/dumps --overwrite-destination
echo "Restore complete. You can now run 'docker-compose up'"
