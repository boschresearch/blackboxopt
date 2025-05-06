# BOHB Optimizer

BOHB performs robust and efficient hyperparameter optimization at scale by combining the
speed of Hyperband searches with the guidance and guarantees of convergence of Bayesian
Optimization.
Instead of sampling new configurations at random, BOHB uses kernel density estimators to
select promising candidates.

This implementation is meant to supersede the initial release of
[HpBandSter](https://github.com/automl/HpBandSter/).


## Fidelities

Here you can calculate the fidelity schedule resulting from BOHB's hyper-parameters:

<script>
function calculateFidelitiesBOHB() {
    const min_fidelity = document.getElementById('minFidelityBOHB').value;
    const max_fidelity = document.getElementById('maxFidelityBOHB').value;
    const eta = document.getElementById('etaBOHB').value;

    const max_num_stages = 1 + Math.floor(
        Math.log(max_fidelity / min_fidelity) / Math.log(eta)
    );
    const num_configs_first_stage = Math.ceil(Math.pow(eta, max_num_stages - 1));
    const num_configs_per_stage = Array.from({ length: max_num_stages }, (_, i) =>
        Math.floor(num_configs_first_stage / Math.pow(eta, i))
    );
    const fidelities_per_stage = Array.from({ length: max_num_stages }, (_, i) =>
        max_fidelity / Math.pow(eta, max_num_stages - 1 - i)
    );

    document.getElementById('fidelitiesBOHB').innerHTML = `Fidelities: ${fidelities_per_stage}`;
}
</script>
<table>
    <tr>
        <td>min_fidelity</td>
        <td><input type="text" id="minFidelityBOHB"></td>
    </tr>
    <tr>
        <td>max_fidelity</td>
        <td><input type="text" id="maxFidelityBOHB"></td>
    </tr>
    <tr>
        <td>eta</td>
        <td><input type="text" id="etaBOHB"></td>
    </tr>
    <tr>
        <td></td>
        <td><button onclick="calculateFidelitiesBOHB();">Calculate</button></td>
    </tr>
</table>
<p id="fidelitiesBOHB"></p>


## Reference

::: blackboxopt.optimizers.bohb
