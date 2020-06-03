# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

all: build

.gen-sidebar.yml: sidebar.yml
	./sidebar.sh

.gen-site: site theme .gen-sidebar.yml
	bundle exec jekyll build

build: .gen-site

run: site theme .gen-sidebar.yml
	bundle exec jekyll server --watch

clean:
	rm -rf .gen-site
