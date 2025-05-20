import shap
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict, shap.sample(X_train, 100))

    def explain(self, X, plot=False, save_path=None):
        """Generate SHAP explanations."""
        shap_values = self.explainer.shap_values(X)
        if plot:
            shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
        return shap_values