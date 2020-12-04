# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

all: build

build/_includes: gen_sidebar.py theme/_includes
	rm -rf build/_includes
	mkdir -p build/_includes
	cp theme/_includes/* build/_includes
	./gen_sidebar.py > build/_includes/sidebar_generated.html

build-site: build/_includes site theme
	bundle exec jekyll build

build: build-site

run: build/_includes site theme
	bundle exec jekyll server --watch

clean:
	rm -rf build
