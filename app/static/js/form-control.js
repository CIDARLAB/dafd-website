//Generate metrics
function populateMetrics(){

    var mode = document.getElementById("mode_select");
    var metrics = document.getElementById("metrics_select");
    var modeSelected = mode.options[mode.selectedIndex].value;

    if (modeSelected=='classification')
    {
        metrics.options.length=0;
        metrics.options[0] = new Option('Accuracy', 'accuracy');
        metrics.options[1] = new Option('Precision', 'precision');
        metrics.options[2] = new Option('Recall', 'recall');
        metrics.options[3] = new Option('F1 Score', 'f1');
        metrics.options[4] = new Option('ROC-AUC', 'roc_auc');
    }
    else if (modeSelected=='regression')
    {
        metrics.options.length=0;
        metrics.options[0] = new Option('MSE', 'neg_mean_squared_error');
        metrics.options[1] = new Option('MAE', 'neg_mean_absolute_error');
        metrics.options[2] = new Option('R-squared', 'r2');
    }
}

function activateFold() {
    var cv = document.getElementById("cv_select").selectedIndex;
    var tune = document.getElementById("hp_select").selectedIndex;
    if (cv == 0 || tune != 2) {
        document.getElementById("fold_text").disabled = false;
    }
    else {
        document.getElementById("fold_text").disabled = true;
    }
    if (tune != 2) {
        document.getElementById("network-setting").innerHTML = "Split with comma (,) for multiple values for each parameter";
    }
    else {
        document.getElementById("network-setting").innerHTML = "Enter a value for each parameter";
    }
}

function validityForm() {

    var str = '';
    var err = 0;
    var target = document.getElementById('target_select').value;
    if (target == 'empty') {
        str += 'No target selected. Please pick a target variable!\n';
        err++;
        //alert('No target selected. Please pick a target variable!');
        //event.preventDefault();
    }

    var model_name = document.getElementById('model-name').value;
    if (model_name == '') {
        str += 'Enter a model name!\n';
        err++;
        //alert('Enter a model name!');
        //event.preventDefault();
    }

    var holdout = document.getElementById('holdout').value;
    //if (holdout != '') {
        holdout = parseFloat(holdout)
        if (isNaN(holdout) || holdout <= 0 || holdout > 100) {
            str += 'Enter a correct number of holdout size (must be between 0 - 100%)!\n';
            err++;
            //alert('Enter a correct number for the holdout size (must be between 0.0 - 1.0)!');
            //event.preventDefault();
        }
    //}

    var cv = document.getElementById("cv_select").selectedIndex;
    var tune = document.getElementById("hp_select").selectedIndex;
    if (cv == 1 || tune == 1) {
        var fold = parseInt(document.getElementById('fold_text').value);
        if (isNaN(fold) || fold < 1 || fold > 10) {
            str += 'Enter a correct number of fold (must be between 1 - 10)!\n';
            err++;
            //alert('Enter a correct number for the number of fold (must be between 1 - 10)!');
            //event.preventDefault();
        }
        /*else if (fold > 3) {
            confirm('Entering fold > 3 requires longer time to finish the hyper-parameter tuning! Do you still want to proceed?');
        }*/
    }

    if (err>0) {
        alert(str);
        event.preventDefault();
    }
    else if (tune == 0 && fold > 3) {
        var proceed = confirm('Entering fold > 3 requires longer time to finish the hyper-parameter tuning! Do you still want to proceed?');
        if (!proceed) {
            event.preventDefault();
        }
    }
}

function validityBackward() {

    var value1 = document.getElementById('dropSizeCheck')
    var value2 = document.getElementById('genRateCheck')

    if(!value1.checked && !value2.checked) {
        alert("Either 'droplet diameter' or 'generation rate' parameter has to be filled!");
        event.preventDefault();
    }
}
