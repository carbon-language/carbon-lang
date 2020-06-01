# Carbon: Jekyll

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Jekyll is used to translate md documentation into html.

Important pieces are:

- `_config.yml`: the Jekyll config.
- `site`: the combination of `theme` and Carbon pages used to build the actual
  site.
- `theme`: resources from the Jekyll theme we use.

## Installing tooling

See [Jekyll install instructions](https://jekyllrb.com/docs/installation/)

## Testing changes

Run `make run` and go to `localhost:4000`.

## Pushing changes

Jekyll changes are autopushed by `/.github/workflows/publish-docs.yaml`.
