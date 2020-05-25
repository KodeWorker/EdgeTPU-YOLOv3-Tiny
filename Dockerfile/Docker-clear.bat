@echo off
::remove all containers
FOR /f "tokens=*" %%i IN ('docker ps -aq') DO docker rm %%i
::remove all images
FOR /f "tokens=*" %%i IN ('docker images --format "{{.ID}}"') DO docker rmi %%i