# MNIST recognition with sk-learn, Docker and Flask

Small dockerized python script that solves the well known MNIST digit recognition problem. 
Shows solvers basic knowledge of Docker, web applications and machine learning. 

## Documentation

### Run locally 
sudo docker run -p 5000:5000 -tagname

### Run with docker-compose
docker-compose up

### How to use API

When calling the API for example with localhost:3000/13
the prediction for the dataset at index 13 of the MNIST test split will be returned
.
.
