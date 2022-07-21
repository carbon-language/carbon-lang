-- Part of the Carbon Language project, under the Apache License v2.0 with LLVM
-- Exceptions. See /LICENSE for license information.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function Link(el)
  el.target = string.gsub(el.target, "^([^//].*).md", "%1.html")
  el.target = string.gsub(el.target, "README.html", "index.html")
  el.target = string.gsub(el.target, "^((?!http).*)/$", "%1/index.html")
  el.target = string.gsub(el.target, "^/", "https://github.com/carbon-language/carbon-lang/tree/trunk/")
  return el
end
