# Website

This website is built using [Docusaurus 2](https://docusaurus.io/), a modern static website generator.

### Installation

```
$ npm install
```

### Local development

```
$ npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```
$ npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

Using SSH:

```
$ USE_SSH=true npm run deploy
```

Not using SSH:

```
$ GIT_USER=<Your GitHub username> npm run deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.

### Setting up GitHub Actions auto-deployment

The [deployment workflow](../.github/workflows/deploy-website.yaml) is configured to deploy to `carbon-language.github.io` using [deploy keys](https://docs.github.com/en/developers/overview/managing-deploy-keys).
(Personal access tokens can also be used via [`personal_token`](https://github.com/peaceiris/actions-gh-pages#%EF%B8%8F-deploy-to-external-repository-external_repository).)

To configure the required deploy keys:

1. Create a new SSH key pair according to the [GitHub docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key).
2. Add the private key to the "Repository secrets" section of repository which runs the workflow, name it `ACTIONS_DEPLOY_KEY`.
3. Add the public key to the "Deploy keys" section of the target deployment repository and enable "Allow write access".
4. The next time the GitHub Actions workflow runs, it will use these keys to deploy the website.
