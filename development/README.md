<br>

## huggingface.co

This repository hosts the programs of a [huggingface.co](https://huggingface.co) `gradio` graphical user interface.  The interface enables interactions with a token classification model.  Within [https://huggingface.co](https://huggingface.co), we create a [`space`](https://huggingface.co/docs/hub/spaces) with the settings

- [x] Space
- [x] Gradio
- [x] Public

<br>

Of import:

* [data](../src/data)
  * <abbr title="The model artefacts of the best model.">model/</abbr>
  * architecture.json
* [app.py](../app.py)
* [requirements.txt](../requirements.txt)


<br>
<br>


## Development Environment

The outlined remote environment is used build and test [https://huggingface.co](https://huggingface.co) Space `gradio` applications.

### Remote Development

The remote development environment requires

* [Dockerfile](../.devcontainer/Dockerfile)
* [requirements.txt](../.devcontainer/requirements.txt)

The image is built via the command

```shell
docker build . --file .devcontainer/Dockerfile -t interact
```

On success, the output of

```shell
docker images
```

should include

<br>

| repository | tag    | image id | created  | size     |
|:-----------|:-------|:---------|:---------|:---------|
| interact   | latest | $\ldots$ | $\ldots$ | $\ldots$ |


<br>

Subsequently, a container, i.e., an instance, of the image `interact` is launched via variations of:

<br>

```shell
docker run --rm --gpus all -i -t 
  -p 7860:7860 -p 8000:8000 -w /app 
    --mount type=bind,src="$(pwd)",target=/app 
      -v ~/.aws:/root/.aws interact
```

<br>

Herein, `-p 7860:7860` maps the host port `7860` to container port `7860`.  Note, the container's working environment, i.e., -w, must be inline with this project's top directory.  Additionally

* --rm: [automatically remove container](https://docs.docker.com/engine/reference/commandline/run/#:~:text=a%20container%20exits-,%2D%2Drm,-Automatically%20remove%20the)
* -i: [interact](https://docs.docker.com/engine/reference/commandline/run/#:~:text=and%20reaps%20processes-,%2D%2Dinteractive,-%2C%20%2Di)
* -t: [tag](https://docs.docker.com/get-started/02_our_app/#:~:text=Finally%2C%20the-,%2Dt,-flag%20tags%20your)
* -p: [publish the container's port/s to the host](https://docs.docker.com/engine/reference/commandline/run/#:~:text=%2D%2Dpublish%20%2C-,%2Dp,-Publish%20a%20container%E2%80%99s)
* --mount type=bind: [a bind mount](https://docs.docker.com/engine/storage/bind-mounts/#syntax)
* -v: [volume](https://docs.docker.com/engine/storage/volumes/)

<br>

Get the name of the running instance of ``interact`` via:

```shell
docker ps --all
```

<br>
<br>

## References

* [Spaces GitHub Actions](https://huggingface.co/docs/hub/spaces-github-actions)
* [huggingface.co repository](https://huggingface.co/docs/huggingface_hub/guides/repository)
* [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker#docker-spaces): [Demonstration](https://huggingface.co/docs/hub/spaces-sdks-docker-first-demo)
* [Switching User in Docker Image or Container](https://www.baeldung.com/linux/docker-image-container-switch-user)

<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>
