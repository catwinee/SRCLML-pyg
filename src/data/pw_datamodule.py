from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule

from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkLoader, LinkNeighborLoader

from components.load_sub_graphs import load_from_json


class PWDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        batch_size: int = 32,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        # TODO: 这个有什么用?
        # self.data_train: Optional[Dataset] = None
        # self.data_val: Optional[Dataset] = None
        # self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed.
        """
        return

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        invoke_subgraph, app_tag_subgraph, api_tag_subgraph = load_from_json()
        print(invoke_subgraph)

        invoke_split = RandomLinkSplit(
            num_val=0,
            num_test=0.2,
            disjoint_train_ratio=0,
            edge_types=("app", "invoke", "api"),
        )
        self.invoke_train_data, _, self.invoke_test_data = invoke_split(invoke_subgraph)

        app_tag_split = RandomLinkSplit(
            num_val=0,
            num_test=0.2,
            disjoint_train_ratio=0,
            edge_types=("app", "app_has_tag", "app_tag"),
        )
        self.pt_train_data, _, self.pt_test_data = app_tag_split(app_tag_subgraph)

        api_tag_split = RandomLinkSplit(
            num_val=0,
            num_test=0.2,
            disjoint_train_ratio=0,
            edge_types=("api", "api_has_tag", "api_tag"),
        )
        self.it_train_data, _, self.it_test_data = api_tag_split(api_tag_subgraph)

    def train_dataloader(self):
        # TODO: 一次返回三个loader？
        return LinkNeighborLoader(
            data=self.invoke_train_data,
            num_neighbors=[30] * 2,
            batch_size=self.batch_size_per_device,
            shuffle=True,
        ), LinkNeighborLoader(
            data=self.pt_train_data,
            num_neighbors=[30] * 2,
            batch_size=self.batch_size_per_device,
            shuffle=True,
        ), LinkNeighborLoader(
            data=self.it_train_data,
            num_neighbors=[30] * 2,
            batch_size=self.batch_size_per_device,
            shuffle=True,
        )

    def val_dataloader(self) -> None:
        # no validation
        return

    def test_dataloader(self):
        return LinkNeighborLoader(
            data=self.invoke_test_data,
            num_neighbors=[30] * 2,
            batch_size=self.batch_size_per_device,
            shuffle=True,
        ), LinkNeighborLoader(
            data=self.pt_test_data,
            num_neighbors=[30] * 2,
            batch_size=self.batch_size_per_device,
            shuffle=True,
        ), LinkNeighborLoader(
            data=self.it_test_data,
            num_neighbors=[30] * 2,
            batch_size=self.batch_size_per_device,
            shuffle=True,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    module = PWDataModule()
    module.setup()
    # _ = module.train_dataloader()
    print(module)
