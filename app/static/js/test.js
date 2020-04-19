$("#submit").click(function() {
    var question = $("#question").val();

    $.ajax({
        url: '/get_intent_nlp',
        dataType: "json",
        contentType: "application/json",
        data: JSON.stringify({
            'question':question,
        }),
        type: 'POST',
        success: function(response) {
            $("#section_json_output").show();
            var data = response;
            $("#json_output").html(data['intents']);
            $("#inference_time").html(data['inference_time'])
        }
    });
});