docker pull tensorflow/tensorflow:1.11.0
docker build -f Dockerfile/Dockerfile -t edgetpu/host .

docker rmi edgetpu/exe
docker build --no-cache -f Dockerfile/DockerfileExe -t edgetpu/exe .