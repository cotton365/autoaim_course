(function () {
  document.querySelectorAll('.quiz').forEach(function (quiz) {
    var answer = quiz.dataset.answer;
    var feedback = quiz.querySelector('.feedback');
    var buttons = quiz.querySelectorAll('button[data-choice]');
    buttons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        buttons.forEach(function (b) { b.classList.remove('correct', 'wrong'); });
        var ok = btn.dataset.choice === answer;
        btn.classList.add(ok ? 'correct' : 'wrong');
        feedback.className = 'feedback ' + (ok ? 'ok' : 'no');
        feedback.textContent = (ok ? '✓ 正确。' : '✗ 再想想。') + (btn.dataset.why || '');
      });
    });
  });
})();
