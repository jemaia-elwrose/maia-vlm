# Sources: test-clips

Short clips, one per check, used to test maia-vlm locally. **No clip is stored in or
distributed with this repository**; only their names, the test cases (`cases.yaml`) and the
results are. Licences are as stated by each dataset's publisher; the underlying footage of
several datasets remains subject to third-party copyright.

| clip | dataset | source | licence (as published) | redistributed here |
|---|---|---|---|---|
| smoke_026.mp4 | HMDB51, class "smoke" | Hugging Face `mteb/HMDB51` | CC BY 4.0 (compilation); clips subject to third-party copyright | no |
| home_fall.mp4 | Le2i Fall Detection | Kaggle `tuyenldvn/falldataset-imvia` | CC BY-NC-SA 4.0 per the ImViA / UBFC release (Kaggle lists "Unknown") | no |
| violence.mp4 | Real Life Violence Situations | Kaggle `mohamedmustafa/real-life-violence-situations-dataset` | CC BY 4.0 per IEEE DataPort; YouTube clips subject to third-party copyright | no |
| shoplifting_07.mp4 | Retail Shoplifting | Kaggle `kipshidze/shoplifting-video-dataset` (Mendeley `xrj39j9633`) | CC BY 4.0 per Mendeley Data (Kaggle lists "Unknown") | no |
| PPE_Part1.mp4 | PPE video sets | Kaggle `hinepo/video-example-for-ppe-red-zone` / `slimese/ppe-video` | none declared (all rights reserved by default) | no |
| line_waiting_720p.mp4 | none | recorded test footage | none declared | no |
| Helmet.mp4 | none | recorded test footage | none declared | no |
| jewellery-phone-1.mp4 | none | synthetic (generated) clip | none declared | no |
| forklift_11s_720p.mp4 | none | recorded test footage | none declared | no |

Clips were trimmed to 10-15 s around the event and converted to H.264 MP4 (720p where
downscaled). Non-commercial and undeclared licences make these clips suitable for local
evaluation only, never for training a model that is sold.