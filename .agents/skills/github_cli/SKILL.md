---
name: GitHub CLI usage
description:
    Instructions for using the `gh` command to query and inspect GitHub state
    safely.
---

# GitHub CLI usage

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This skill provides instructions for using the GitHub CLI (`gh`) to query,
inspect, and search GitHub state (issues, pull requests, repositories) for the
Carbon project.

## Safety First: Read-Only Usage

> [!IMPORTANT] AI assistants MUST NOT use the `gh` tool to modify any GitHub
> project state. Do NOT run commands that create, edit, delete, label, comment
> on, or merge issues, pull requests, releases, or any other resources.

### Allowed Verbs

-   `list`
-   `view`
-   `search`
-   `status`
-   `api` (Only with `GET` requests)

### Prohibited Verbs

-   `create`
-   `edit`
-   `delete`
-   `merge`
-   `reopen`
-   `close`
-   `comment`
-   `label`

## Repository Configuration

The `gh` tool interacts with a default repository when run within a local check
out. For this project, the default repository is expected to be
`carbon-language/carbon-lang`.

### Verifying Default Repository

To verify the current default repository configuration:

```bash
gh repo view
```

The output should indicate the repository is `carbon-language/carbon-lang`.

### Correcting Misconfigurations

If the default repository is misconfigured (for example, pointing to a personal
fork or a different repository), the human operator must correct it.

> [!IMPORTANT] AI Assistants MUST NOT attempt to mutate `gh` configuration or
> run commands that change the default repository (such as
> `gh repository set-default`).

Instruct the human operator to run the following command to select the correct
default repository:

```bash
gh repo set-default
```

The operator will be prompted to select the correct repository (e.g.,
`carbon-language/carbon-lang`) from the available remotes.

## Common Query Commands

### Issues

-   **List issues**: `gh issue list`
-   **View specific issue**: `gh issue view <number>`
-   **Search issues**: `gh issue search "<query>"`
    -   Example: `gh issue search "crash" --state open`

### Pull Requests

-   **List PRs**: `gh pr list`
-   **View specific PR**: `gh pr view <number>`
-   **View PR diff**: `gh pr diff <number>`
-   **Check PR status**: `gh pr status`

### Search

-   **Search code**: `gh search code "<query>"`
-   **Search repositories**: `gh search repos "<query>"`

## Advanced Usage: GitHub API

For queries that are not supported by standard `gh` commands, you can use the
`gh api` command to query the GitHub REST or GraphQL APIs.

### REST API

Query the REST API using paths relative to the API root.

-   **List contributors**:

    ```bash
    gh api repos/carbon-language/carbon-lang/contributors
    ```

-   **List issue comments**:

    ```bash
    gh api repos/carbon-language/carbon-lang/issues/<issue_number>/comments
    ```

### GraphQL API

For complex queries, use GraphQL to fetch exactly the data needed.

-   **Get repository information**:

    ```bash
    gh api graphql -f query='
      query {
        repository(owner: "carbon-language", name: "carbon-lang") {
          description
          stargazerCount
        }
      }
    '
    ```

### Pagination

Use the `--paginate` flag to automatically fetch all pages of results.

```bash
gh api --paginate repos/carbon-language/carbon-lang/issues
```

### Filtering and Formatting

Use `--json` to request JSON output, and `--jq` or `--template` to filter or
format the results.

-   **List PR titles and authors**:

    ```bash
    gh pr list --json title,author --jq '.[] | "\(.title) by \(.author.login)"'
    ```

-   **Format with Go templates**:

    ```bash
    gh issue list --template '{{range .}}{{.number}} - {{.title}}{{"\n"}}{{end}}'
    ```

## Documentation References

-   **GitHub CLI Manual**:
    [cli.github.com/manual](https://cli.github.com/manual/)
-   **GitHub REST API Documentation**:
    [docs.github.com/en/rest](https://docs.github.com/en/rest)
-   **GitHub GraphQL API Documentation**:
    [docs.github.com/en/graphql](https://docs.github.com/en/graphql)
