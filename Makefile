.PHONY: help install

help:
	@(cat $(firstword $(MAKEIFLE_LIST)))

install:
	pip install -r requirements.txt

