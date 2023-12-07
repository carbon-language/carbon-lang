# Testing workflows

We keep around an `action-test` branch in carbon-lang, which can be used to test
triggers with `push:` configurations. For example:

```
on:
  push:
    branches: [action-test]
```
