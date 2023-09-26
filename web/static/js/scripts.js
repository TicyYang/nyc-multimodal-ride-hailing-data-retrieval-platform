/*!
* Start Bootstrap - Clean Blog v6.0.9 (https://startbootstrap.com/theme/clean-blog)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-clean-blog/blob/master/LICENSE)
*/



window.addEventListener('DOMContentLoaded', () => {
    let scrollPos = 0;
    const mainNav = document.getElementById('mainNav');
    const headerHeight = mainNav.clientHeight;
    window.addEventListener('scroll', function () {
        const currentTop = document.body.getBoundingClientRect().top * -1;
        if (currentTop < scrollPos) {
            // Scrolling Up
            if (currentTop > 0 && mainNav.classList.contains('is-fixed')) {
                mainNav.classList.add('is-visible');
            } else {
                console.log(123);
                mainNav.classList.remove('is-visible', 'is-fixed');
            }
        } else {
            // Scrolling Down
            mainNav.classList.remove(['is-visible']);
            if (currentTop > headerHeight && !mainNav.classList.contains('is-fixed')) {
                mainNav.classList.add('is-fixed');
            }
        }
        scrollPos = currentTop;
    });
})


// 日期選擇器開始
import AirDatepicker from 'air-datepicker';
import { createPopper } from '@popperjs/core';

new AirDatepicker('#el', {
    container: '#scroll-container',
    visible: true,
    position({ $datepicker, $target, $pointer, done }) {
        let popper = createPopper($target, $datepicker, {
            placement: 'top',
            modifiers: [
                {
                    name: 'flip',
                    options: {
                        padding: {
                            top: 64
                        }
                    }
                },
                {
                    name: 'offset',
                    options: {
                        offset: [0, 20]
                    }
                },
                {
                    name: 'arrow',
                    options: {
                        element: $pointer
                    }
                }
            ]
        })

        /*
         Return function which will be called when `hide()` method is triggered,
         it must necessarily call the `done()` function
             to complete hiding process 
        */
        return function completeHide() {
            popper.destroy();
            done();
        }
    }
})

//日期選擇器結束