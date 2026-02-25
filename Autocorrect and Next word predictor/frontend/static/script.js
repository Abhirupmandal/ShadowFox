const txt = document.getElementById("inputText");

txt.addEventListener("keyup", e => {

    if (e.key !== " ") return;

    fetch("/predict", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({text: txt.value})
    })
    .then(r => r.json())
    .then(d => {

        document.getElementById("corrected").innerText =
            "Autocorrect: " + d.corrected;

        document.getElementById("predictions").innerText =
            "Next Words: " + d.predictions.join(", ");
    });

});