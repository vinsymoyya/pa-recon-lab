# Security policy

This is development research software; there is no long-term supported release yet. Use the latest reviewed source in an isolated research environment.

Report a suspected vulnerability through GitHub's private vulnerability reporting for this repository when enabled, or use a contact option on [the maintainer profile](https://github.com/vinsymoyya) to arrange private disclosure. Include the affected revision, reproducible steps, impact, and a proposed fix if available. Do not publish credentials, private scans, or exploit details in public issues before coordination.

Treat external checkpoints, HDF5 files, and notebook code as untrusted inputs. The CLI uses PyTorch's restricted `weights_only=True` loader. Check artifact provenance and integrity before use. Dataset licenses and research/medical validity are documented separately in the model card and data audit.
