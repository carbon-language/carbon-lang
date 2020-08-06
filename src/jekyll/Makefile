# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

all: build

.gen-site: site theme
	bundle exec jekyll build

build: .gen-site

run: site theme
	bundle exec jekyll server --watch

clean:
	rm -rf .gen-site
