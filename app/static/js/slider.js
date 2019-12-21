function updateIntInput(val, name, slider) {
    document.getElementById(name).value = val;
    document.getElementById(slider).classList.add('slider-activated');
    document.getElementById(slider).classList.remove('slider-warning');
}

function updateFloatInput(val, name, slider) {
    document.getElementById(name).value = val/100;
    document.getElementById(slider).classList.add('slider-activated');
    document.getElementById(slider).classList.remove('slider-warning');
}

function updateIntSlider(slider, text) {
    var val = document.getElementById(text).value
    document.getElementById(slider).value = val;
}

function intSliderChange(slider, text, min, max) {
    var val = document.getElementById(text).value
    if(isNaN(val) || val < 0) {
        alert('Please enter a valid and a positive number');
        document.getElementById(slider).classList.add('slider-activated');
        document.getElementById(slider).classList.remove('slider-warning');
        val = min;
    }
    else if(val < min) {
        if (confirm('You are entering out of range value. Do you want to proceed?')) {
            document.getElementById(slider).classList.add('slider-warning');
            document.getElementById(slider).classList.remove('slider-activated');
        }
        else {
            val = min;
        }
    }
    else if(val > max) {
        if (confirm('You are entering out of range value. Do you want to proceed?')) {
            document.getElementById(slider).classList.add('slider-warning');
            document.getElementById(slider).classList.remove('slider-activated');
        }
        else {
            val = max;
        }
    }
    else if(val >= min && val <= max) {
        document.getElementById(slider).classList.add('slider-activated');
        document.getElementById(slider).classList.remove('slider-warning');
    }
    document.getElementById(slider).value = val;
    document.getElementById(text).value = val;
}

function updateFloatSlider(slider, text) {
    var val = document.getElementById(text).value * 100;
    document.getElementById(slider).value = val;
}

function floatSliderChange(slider, text, min, max) {
    var val = document.getElementById(text).value * 100;
    if(isNaN(val) || val < 0) {
        alert('Please enter a valid and a positive number');
        document.getElementById(slider).classList.add('slider-activated');
        document.getElementById(slider).classList.remove('slider-warning');
        val = min;
    }
    else if(val < min) {
        if (confirm('You are entering out of range value. Do you want to proceed?')) {
            document.getElementById(slider).classList.add('slider-warning');
            document.getElementById(slider).classList.remove('slider-activated');
        }
        else {
            val = min;
        }
    }
    else if(val > max) {
        if (confirm('You are entering out of range value. Do you want to proceed?')) {
            document.getElementById(slider).classList.add('slider-warning');
            document.getElementById(slider).classList.remove('slider-activated');
        }
        else {
            val = max;
        }
    }
    else if(val >= min && val <= max) {
        document.getElementById(slider).classList.add('slider-activated');
        document.getElementById(slider).classList.remove('slider-warning');
    }
    document.getElementById(slider).value = val;
    document.getElementById(text).value = val/100;
}

function updateLambda(lambda, gen_rate, flow_rate) { 
    document.getElementById('conc').value =  gen_rate * lambda * 60 / flow_rate
}

function disableSlider(id, targetSlider, targetText) {
    var value = document.getElementById(id)
    if(value.checked) {
        document.getElementById(targetSlider).disabled = false;
        document.getElementById(targetText).disabled = false;
        document.getElementById(targetSlider).classList.add('slider-activated');
        document.getElementById(targetSlider).classList.remove('slider');
    }
    else {
        document.getElementById(targetSlider).disabled = true;
        document.getElementById(targetText).disabled = true;
        document.getElementById(targetSlider).classList.remove('slider-activated');
        document.getElementById(targetSlider).classList.add('slider');
    }
}

function disableCombo(id, targetCombo) {
    var value = document.getElementById(id)
    if(value.checked) {
        document.getElementById(targetCombo).disabled = false;
    }
    else {
        document.getElementById(targetCombo).disabled = true;
    }
}

//not sure what this is for
$('.field-tip').tooltip({
    disabled: true,
    close: function( event, ui ) { $(this).tooltip('disable'); }
});

$('.field-tip').on('click', function () {
    $(this).tooltip('enable').tooltip('open');
});
