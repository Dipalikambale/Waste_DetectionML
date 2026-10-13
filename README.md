# Waste Detection

An end-to-end waste-object detection project using YOLOv5 through Ultralytics,
with dataset ingestion and validation, versioned model artifacts, a Flask
inference app, Docker, and GitHub Actions deployment to AWS ECR and EC2.

## Project layout

```text
.
├── .github/workflows/ci-cd.yml
├── data/waste.yaml
├── deploy/cloudformation.yaml
├── src/waste_detection/
│   ├── app.py
│   ├── artifacts.py
│   ├── config.py
│   ├── ingestion.py
│   ├── training.py
│   └── validation.py
├── tests/
├── Dockerfile
└── requirements*.txt
```

## 1. Prepare a dataset

Use the YOLO detection layout below. Every image with objects should have a
matching `.txt` label file; images with no objects may have no label file.

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each label line is `class_id x_center y_center width height`, with normalized
coordinates in the range `[0, 1]`. Update `data/waste.yaml` to match the actual
class names and class IDs in your labels. Its six starter classes are
`cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`; they are examples,
not labels inferred from an included dataset. Images and trained weights are
not included.

Install the training dependencies and copy a dataset into the project:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m waste_detection.cli ingest D:\path\to\dataset data\waste-dataset
```

The `ingest` command copies the source dataset to the requested destination. For
the destination above, the starter `data/waste.yaml` already points to
`data/waste-dataset`; if you chose another destination, update `path` to its
location relative to `data/waste.yaml` (or use an absolute path), then validate
and train.

### Prepare the public TACO detection dataset

To use TACO instead, run the following command from the project root:

```powershell
python -m waste_detection.cli prepare-taco
```

It downloads the official TACO annotations and the images needed for the
selected categories, then writes a seeded 80/20 train-validation split as
YOLO-format labels under `data/taco-yolo`. The default class mapping groups
TACO annotations into `bottle`, `can`, and `plastic_bag_or_wrapper`; other TACO
categories are excluded from training. The output config is
`data/taco-yolo/data.yaml`. It refuses to overwrite an existing output folder.
The source dataset uses images hosted on Flickr: respect applicable image
licenses and use this prepared copy for local experimentation rather than
redistributing the images.

For an optional small smoke-test dataset, pass `--limit 100`; for a real first
training run, omit the limit to use all matching TACO images. Validate the
converted labels before training:

```powershell
python -m waste_detection.cli validate --data data/taco-yolo/data.yaml
python -m waste_detection.cli train --data data/taco-yolo/data.yaml
```

Training defaults are 30 epochs, 416-pixel images, and batch size 2 for this
CPU-first setup. Expect training time to depend heavily on CPU speed. To try a
short smoke-training run, add `--epochs 1`; to train longer, adjust `--epochs`.
Training uses Ultralytics' pretrained YOLOv5u nano checkpoint `yolov5nu.pt` by
default; it is an up-to-date, smaller starting model suited to CPU experiments.
Override it with `--weights`. Ultralytics handles image resizing, augmentation,
and training preprocessing. Training metrics and run outputs are saved under
`artifacts/runs/`; timestamped weight versions, metadata, and the active
`artifacts/best.pt` are saved under `artifacts/`.

## 2. Run inference

After training (or after providing a compatible checkpoint at
`artifacts/best.pt`), run:

```powershell
python -m waste_detection.app
```

Open `http://localhost:5000`. Upload an image to view bounding boxes and
predicted classes. The JSON API is `POST /api/predict` with a multipart
`image` field. `GET /health` reports whether a model is available and
`GET /ready` returns 503 until the configured weight file exists.

Useful environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `WASTE_MODEL_PATH` | `artifacts/best.pt` | Model checkpoint |
| `WASTE_METADATA_PATH` | `artifacts/metadata.json` | Training metadata |
| `WASTE_CONFIDENCE` | `0.25` | Minimum detection confidence |
| `WASTE_MAX_UPLOAD_MB` | `16` | Maximum uploaded image size |
| `WASTE_HOST` / `WASTE_PORT` | `0.0.0.0` / `5000` | Flask/Gunicorn bind |

## 3. Run tests and Docker

```powershell
python -m pip install -r requirements-test.txt
python -m pip install -e .
python -m pytest
docker build -t waste-detection .
docker run --rm -p 5000:5000 -v "${PWD}\artifacts:/app/artifacts" waste-detection
```

The Docker image runs as a non-root user. Mount a trained `artifacts` directory
at runtime; weights are intentionally not baked into the image.

## 4. AWS ECR and EC2 deployment

The GitHub Actions workflow runs tests on pushes and pull requests. On pushes
to `main`, it builds a CPU inference image, pushes a unique immutable tag to
ECR, then uses AWS Systems Manager (SSM) to update the configured EC2 instance.
The instance pulls from ECR and the model bucket using its instance profile.

1. Create the GitHub Actions OIDC provider in IAM if your account does not
   already have one (`https://token.actions.githubusercontent.com`, audience
   `sts.amazonaws.com`).
2. Deploy `deploy/cloudformation.yaml` in your AWS account with `AllowedIngressCidr`,
   `GitHubOwner`, and `GitHubRepository` parameters. Optionally set
   `GitHubBranch` and `InstanceType`. This creates the restricted GitHub deploy
   role, ECR repository, encrypted/versioned model bucket, and EC2 host.
3. Upload the trained active checkpoint and its metadata to the bucket output
   by the stack:

   ```powershell
   aws s3 cp artifacts\best.pt s3://<ModelBucketName>/models/best.pt
   aws s3 cp artifacts\metadata.json s3://<ModelBucketName>/models/metadata.json
   ```

4. Add the `AWS_DEPLOY_ROLE_ARN` repository secret using the
   `GitHubDeployRoleArn` output. Set these Actions variables from the stack
   outputs: `AWS_REGION`, `ECR_REPOSITORY_URI`, `EC2_INSTANCE_ID`, and
   `MODEL_BUCKET`.
5. Push to `main`. The instance must be able to reach ECR, S3, and SSM endpoints
   (the template provisions a public subnet and internet gateway).

The template opens HTTP port 80 only to the CIDR you specify; keep the instance
and app behind an HTTPS-terminating load balancer or reverse proxy for
production use. Scale-out, Kafka/Kinesis streams, and live camera ingestion are
not provisioned by this starter project; the validated batch dataset flow and
HTTP image inference endpoint are the working entry points.

## Operational notes

- Do not commit datasets, trained weights, AWS credentials, or `.env` files.
- Keep `data.yaml` class ordering exactly aligned with the numeric IDs in label
  files.
- Validation checks split directories, image presence, labels, class IDs, and
  normalized box coordinates before training begins.
- Inference rejects uploads over the configured byte limit and images larger
  than 20 megapixels.
- Artifact metadata records the checkpoint hash, class names, training run, and
  evaluation metrics. Save versioned artifacts outside the container for
  durable model history.
