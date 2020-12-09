build:
	docker build -t scaled-yolov4:1.0 .

run:
	@docker run --gpus all --rm -ti --ipc=host \
	--name scaledYOLOv4 \
	-v $$(pwd):/workspace \
	-v /home/datasets/coco/coco:/coco \
	-w /workspace \
	-e CUDA_VISIBLE_DEVICES="1" \
	-e PYTHONPATH=/workspace \
	scaled-yolov4:1.0 \
	bash