from typing import Optional, Callable, Dict, Union, List, Collection, Any

import numpy as np
from quantus import MPRT, ModelInterface
from quantus.helpers import warn
from tqdm.asyncio import tqdm


class NicePicturesMPRT(MPRT):
    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = False,
        device: Optional[str] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> Union[List[float], float, Dict[str, List[float]], Collection[Any]]:
        """
        See original MPRT docstring.
        """
        # THIS IS WHAT WE ADDED
        self.a_instance_perturbed_output = []
        # THIS IS WHAT WE ADDED

        # Run deprecation warnings.
        warn.deprecation_warnings(kwargs)
        warn.check_kwargs(kwargs)
        self.batch_size = batch_size
        data = self.general_preprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=None,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            softmax=softmax,
            device=device,
        )
        model: ModelInterface = data["model"]  # type: ignore
        # Here _batch refers to full dataset.
        x_full_dataset = data["x_batch"]
        y_full_dataset = data["y_batch"]
        a_full_dataset = data["a_batch"]

        # Results are returned/saved as a dictionary not as a list as in the super-class.
        self.evaluation_scores = {}

        # Get number of iterations from number of layers.
        n_layers = model.random_layer_generator_length
        pbar = tqdm(
            total=n_layers * len(x_full_dataset), disable=not self.display_progressbar
        )
        if self.display_progressbar:
            # Set property to False, so we display only 1 pbar.
            self._display_progressbar = False

        with pbar as pbar:
            for l_ix, (layer_name, random_layer_model) in enumerate(
                model.get_random_layer_generator(order=self.layer_order, seed=self.seed)
            ):
                pbar.desc = f"{l_ix} {layer_name}"
                print(f"{l_ix} {layer_name}")

                if l_ix == 0:

                    # Generate explanations on original model in batches.
                    a_original_generator = self.generate_explanations(
                        model.get_model(), x_full_dataset, y_full_dataset, batch_size
                    )

                    # Compute the similarity of explanations of the original model.
                    self.evaluation_scores["original"] = []
                    for a_batch, a_batch_original in zip(
                        self.generate_a_batches(a_full_dataset), a_original_generator
                    ):
                        for a_instance, a_instance_original in zip(
                            a_batch, a_batch_original
                        ):
                            self.a_instance_perturbed_output.append(a_instance_original)
                            score = self.evaluate_instance(
                                model=model,
                                x=None,
                                y=None,
                                s=None,
                                a=a_instance,
                                a_perturbed=a_instance_original,
                            )
                            # Save similarity scores in a result dictionary.
                            self.evaluation_scores["original"].append(score)
                            pbar.update(1)

                # Skip layers if computing delta.
                if self.skip_layers and (l_ix + 1) < n_layers:
                    continue

                self.evaluation_scores[layer_name] = []

                # Generate explanations on perturbed model in batches.
                a_perturbed_generator = self.generate_explanations(
                    random_layer_model, x_full_dataset, y_full_dataset, batch_size
                )

                # Compute the similarity of explanations of the perturbed model.
                for a_batch, a_batch_perturbed in zip(
                    self.generate_a_batches(a_full_dataset), a_perturbed_generator
                ):
                    for a_instance, a_instance_perturbed in zip(
                        a_batch, a_batch_perturbed
                    ):
                        # THIS IS WHAT WE ADDED
                        self.a_instance_perturbed_output.append(a_instance_perturbed.copy())
                        # THIS IS WHAT WE ADDED

                        score = self.evaluate_instance(
                            model=random_layer_model,
                            x=None,
                            y=None,
                            s=None,
                            a=a_instance,
                            a_perturbed=a_instance_perturbed,
                        )
                        self.evaluation_scores[layer_name].append(score)
                        pbar.update(1)

        if self.return_average_correlation:
            self.evaluation_scores = self.recompute_average_correlation_per_sample()

        elif self.return_last_correlation:
            self.evaluation_scores = self.recompute_last_correlation_per_sample()

        if self.return_aggregate:
            assert self.return_average_correlation or self.return_last_correlation, (
                "Set 'return_average_correlation' or 'return_last_correlation'"
                " to True in order to compute the aggregate evaluation results."
            )
            self.evaluation_scores = [self.aggregate_func(self.evaluation_scores)]

        # Return all_evaluation_scores according to Quantus.
        self.all_evaluation_scores.append(self.evaluation_scores)

        return self.all_evaluation_scores
