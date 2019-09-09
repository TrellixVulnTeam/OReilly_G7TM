IMAGE_NAME=determinedai/tensorflow-object-detection

build:
	docker build -t $(IMAGE_NAME) .
	docker build -f Dockerfile.gpu -t $(IMAGE_NAME)-gpu .
