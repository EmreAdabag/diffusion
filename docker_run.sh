docker run -d --gpus all \
	-v $(pwd):/mount \
	--shm-size=32gb \
	-p 8888:8888 \
diffusionnew
