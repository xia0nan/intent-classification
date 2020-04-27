$("#submit").click(function() {
    var question = $("#question").val();

    $.ajax({
        url: '/get_intent_nlp_clustering',
        dataType: "json",
        contentType: "application/json",
        data: JSON.stringify({
            'question':question,
        }),
        type: 'POST',
        success: function(response) {
            $("#section_json_output").show();
            var data = response;        // JSON response
            $("#json_output").html(JSON.stringify(data));
        }
    });
});

$("#question").on('keypress',function(e) {
    if(e.which == 13) {
        $("#submit").click();
    }
});