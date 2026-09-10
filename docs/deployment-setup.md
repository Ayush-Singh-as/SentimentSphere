# Owner setup: training server and public demo

The owner controls accounts and credentials. Do not paste private keys, passwords,
or access tokens into chat. Send resource identifiers and credential locations.

## SSH training server

Supply a working SSH alias (or hostname, port, username), a writable workspace,
VPN/jump-host requirements, and scheduling/storage limits. Say whether the GPU is
shared or requires a scheduler such as Slurm. Do not assume an interactive login
node is also the permitted training node.

If passwordless `ssh ALIAS` already works locally, no new key is needed. Otherwise,
create a dedicated passphrase-protected key in your own terminal:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/sentimentsphere_gpu -C sentimentsphere-training
ssh-copy-id -i ~/.ssh/sentimentsphere_gpu.pub -p PORT USERNAME@HOST
```

For providers with a key dashboard, upload the `.pub` file there instead. Never
upload the file without `.pub`. Add a local `~/.ssh/config` entry:

```sshconfig
Host sentimentsphere-gpu
    HostName HOST
    User USERNAME
    Port PORT
    IdentityFile ~/.ssh/sentimentsphere_gpu
    IdentitiesOnly yes
    ForwardAgent no
```

If needed, start a local agent with `eval "$(ssh-agent -s)"`, then run:

```bash
ssh-add ~/.ssh/sentimentsphere_gpu
ssh sentimentsphere-gpu
```

Verify the host fingerprint with the provider before accepting the first
connection. Exit the remote shell, then check noninteractive access:

```bash
ssh -o BatchMode=yes sentimentsphere-gpu 'hostname; nvidia-smi; df -h "$HOME"'
```

The assistant can use an available local agent without learning the passphrase.
Tool sandbox/network approvals may still be needed. No agent forwarding is
required for syncing project files and running commands on the server.

## Hugging Face account and destinations

1. Create/verify a [personal account](https://huggingface.co/join).
2. Visit [New Space](https://huggingface.co/new-space). For the Docker deployment,
   choose your account, `sentimentsphere`, Docker/blank template, CPU Basic, public
   visibility, and MIT for the app source. The implementation supplies the image
   and port configuration; an empty Space is not a working deployment.
3. At [New model repository](https://huggingface.co/new), create
   `USERNAME/sentimentsphere-models`, initially private. Do not assume the app's
   MIT source license also licenses every dataset or trained model.

The original plan assumed unrestricted free Docker hosting. Current
[HF documentation](https://huggingface.co/docs/hub/spaces-overview) says Docker
creation needs a paid account plan; CPU Basic itself has no hourly charge. Eligible
free accounts have a Gradio/ZeroGPU alternative. Check account eligibility before
choosing a paid plan. No subscription purchase or paid hardware upgrade is implied
by this setup guide. If the chosen option is unavailable, report that fact so the
deployment design can be adapted.

## Local deployment authentication

Create a fine-grained token called `sentimentsphere-deploy` at
[Access Tokens](https://huggingface.co/settings/tokens), with repository read/write
access limited to the Space and model repository. Follow the
[token scope documentation](https://huggingface.co/docs/hub/security-tokens).
This credential is for uploading code/artifacts, not for the public runtime.

From the project root, in your own terminal:

```bash
.venv/bin/hf auth login
.venv/bin/hf auth whoami
```

Choose the interactive token-paste option during login; Git credential storage is
optional and unnecessary for Hub-client uploads. The second command verifies the
account without displaying the token. Existing locally cached HF authentication is
sufficient; duplicating it in `.env` is not required. See the
[CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli).

## Runtime access

For private model downloads, make a separate `sentimentsphere-runtime` fine-grained
token with read-only access to the model repository. Put it in the Space Settings
as a **secret** named `HF_TOKEN`. Do not give the web app your deployment write
token. Model IDs and revision pins are nonsecret configuration; the deployment
implementation will provide their exact variable names. See
[Docker environment configuration](https://huggingface.co/docs/hub/spaces-sdks-docker).

Report only:

```text
SSH alias: sentimentsphere-gpu
Remote workspace: /your/writable/path
GPU scheduling restrictions: ...
HF username: ...
Space ID: USERNAME/sentimentsphere
Model repository ID: USERNAME/sentimentsphere-models
Local HF login: completed
Runtime read token: configured
```

Training, artifact validation, deployment, and external smoke tests remain separate
implementation steps. A successful login or repository creation does not establish
model quality or a functioning public demo.
