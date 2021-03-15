# Jekyll

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Jekyll is used to translate md documentation into html.

Important pieces are:

-   `_config.yml`: the Jekyll config.
-   `site`: the combination of `theme` and Carbon pages used to build the actual
    site.
-   `theme`: resources from the Jekyll theme we use.

## Installing tooling

See [Jekyll install instructions](https://jekyllrb.com/docs/installation/). This
will look like:

```bash
brew install ruby
gem install --user-install bundler jekyll
```

Then add `$HOME/.gem/ruby/X.X.0/bin` to the `PATH` environment variable; the
required path will be printed in `gem` output.

Finally, install from the `Gemfile` (this must be run in this directory):

```bash
bundle install
```

`bundle update` will update versions.

## Testing changes

Manual testing of the site can be done two ways:

-   `bazel build :jekyll_site.tgz` can be used to view file contents.
-   `bazel run :serve` can be used to run a server.

## Pushing changes

`bazel run :publish`

Note this updates the live site.
